import subprocess, time, gc, os, sys
class Setup:
    def __init__(self) -> None:
        start_time = time.time()
        print_subprocess = False
        use_xformers_for_colab = True
        try:
            ipy = get_ipython()
        except:
            ipy = 'could not get_ipython'
        if 'google.colab' in str(ipy):
            print("..setting up environment")

            # weird hack
            # import torch
            
            all_process = [
                ['pip', 'install', 'omegaconf', 'einops==0.4.1', 'pytorch-lightning==1.7.7', 'torchmetrics', 'transformers', 'safetensors', 'kornia'],
                ['git', 'clone', 'https://github.com/deforum-art/deforum-stable-diffusion'],
                ['pip', 'install', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq','scikit-learn','torchsde','open-clip-torch','numpngw'],
            ]
            for process in all_process:
                running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                if print_subprocess:
                    print(running)
            with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
                f.write('')
            sys.path.extend([
                'deforum-stable-diffusion/',
                'deforum-stable-diffusion/src',
            ])
            if use_xformers_for_colab:

                print("..installing triton and xformers")

                all_process = [['pip', 'install', 'triton==2.0.0.dev20221202', 'xformers==0.0.16rc424']]
                for process in all_process:
                    running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                    if print_subprocess:
                        print(running)
        else:
            sys.path.extend([
                'src'
            ])
        end_time = time.time()
        print(f"..environment set up in {end_time-start_time:.0f} seconds")