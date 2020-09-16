# from processors.ddp_apex_processor import DDPApexProcessor
from processors.ddp_mix_processor import DDPMixProcessor

# python -m torch.distributed.launch --nproc_per_node=2 --master_port 50003 main.py

if __name__ == '__main__':
    processor = DDPMixProcessor(cfg_path="config/focs.yaml")
    processor.run()
