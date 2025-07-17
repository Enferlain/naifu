import torch
import lightning as pl
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..common.utils import get_class
from .sdxl_model import StableDiffusionModel
from enhanced_llm_to_sdxl_adapter import EnhancedLLMToSDXLAdapter

class LLMAdapterFinetune(StableDiffusionModel):
    def __init__(self, model_path: str, config: OmegaConf, device: torch.device = torch.device("cpu")):
        # This will initialize the SDXL parts (VAE, UNet, Scheduler) from the base class
        super().__init__(model_path, config, device)

        # The base models are frozen by default in naifu, we only need to manage our new components.
        weight_dtype = torch.float16 if self.config.lightning.precision == "16-mixed" else torch.bfloat16 if self.config.lightning.precision == "bf16-mixed" else torch.float32

        # Load the LLM and our adapter, which are specific to this training module
        self.llm_tokenizer = AutoTokenizer.from_pretrained(config.model.llm_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            config.model.llm_path,
            torch_dtype=weight_dtype,
            output_hidden_states=True
        ).to(device)
        self.llm_model.requires_grad_(False)
        self.llm_model.eval()

        self.adapter = EnhancedLLMToSDXLAdapter(
            llm_dim=self.llm_model.config.hidden_size,
            **config.model.get("adapter_params", {})
        ).to(device)
        # Only the adapter is trainable
        self.adapter.train()

    def encode_batch(self, batch):
        """
        Overrides the default conditioning method to use the LLM and adapter
        instead of the standard CLIP text encoders.
        """
        captions = batch.get("captions", "")
        
        # Get LLM hidden states
        with torch.no_grad():
            inputs = self.llm_tokenizer(
                captions, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=self.llm_tokenizer.model_max_length
            ).to(self.target_device)
            llm_outputs = self.llm_model(**inputs)
            # Use the last hidden state
            llm_hidden_states = llm_outputs.hidden_states[-1].to(next(self.adapter.parameters()).dtype)

        # Use the adapter to get SDXL-compatible conditioning
        cross_attn_cond, pooled_cond = self.adapter(llm_hidden_states)

        return {
            "crossattn": cross_attn_cond,
            "vector": pooled_cond
        }

    # The `forward` method is inherited from StableDiffusionModel and will work correctly
    # with our custom `encode_batch` method. No need to override it.

def setup(fabric: pl.Fabric, config: OmegaConf) -> tuple:
    # This setup is now much cleaner and aligns with other naifu modules
    model = LLMAdapterFinetune(
        model_path=config.trainer.model_path, 
        config=config, 
        device=fabric.device
    )
    
    dataset_class = get_class(config.dataset.get("name", "data.bucket.AspectRatioDataset"))
    dataset = dataset_class(
        batch_size=config.trainer.batch_size,
        rank=fabric.global_rank,
        dtype=torch.float32,
        **config.dataset,
    )
    dataloader = dataset.init_dataloader()
    
    # Set up optimizer to train only the adapter parameters
    params_to_optim = [{'params': model.adapter.parameters()}]

    optimizer = get_class(config.optimizer.name)(
        params_to_optim, **config.optimizer.params
    )
    
    scheduler = None
    if config.get("scheduler"):
        scheduler = get_class(config.scheduler.name)(
            optimizer, **config.scheduler.params
        )
    
        # Fabric setup
    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    return model, dataset, dataloader, optimizer, scheduler
