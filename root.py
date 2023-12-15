class Root:
    def __init__(self) -> None:
        self.models_path = "models" #@param {type:"string"}
        self.configs_path = "configs" #@param {type:"string"}
        self.output_path = "outputs" #@param {type:"string"}
        self.deforum_images_path = "outputs/deforum_images"  # @param {type:"string"}
        self.deforum_videos_path = "outputs/deforum_videos"  # @param {type:"string"}
        self.mount_google_drive = True #@param {type:"boolean"}
        self.models_path_gdrive = "/content/drive/MyDrive/AI/models" #@param {type:"string"}
        self.output_path_gdrive = "/content/drive/MyDrive/AI/StableDiffusion" #@param {type:"string"}

        #@markdown **Model Setup**
        self.map_location = "cuda" #@param ["cpu", "cuda"]
        self.model_config = "v1-inference.yaml" #@param ["custom","v2-inference.yaml","v2-inference-v.yaml","v1-inference.yaml"]
        self.model_checkpoint =  "Protogen_V2.2.ckpt" #@param ["custom","v2-1_768-ema-pruned.ckpt","v2-1_512-ema-pruned.ckpt","768-v-ema.ckpt","512-base-ema.ckpt","Protogen_V2.2.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
        self.custom_config_path = "" #@param {type:"string"}
        self.custom_checkpoint_path = "" #@param {type:"string"}
        
    def getLocals(self):
        return {
            "models_path": self.models_path,
            "configs_path": self.configs_path,
            "output_path": self.output_path,
            "deforum_images_path": self.deforum_images_path,
            "deforum_videos_path": self.deforum_videos_path,
            "mount_google_drive": self.mount_google_drive,
            "models_path_gdrive": self.models_path_gdrive,
            "output_path_gdrive": self.output_path_gdrive,
            "map_location": self.map_location,
            "model_config": self.model_config,
            "model_checkpoint": self.model_checkpoint,
            "custom_config_path": self.custom_config_path,
            "custom_checkpoint_path": self.custom_checkpoint_path,
        }