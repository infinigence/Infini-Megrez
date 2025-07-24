from vllm import LLM
from vllm import SamplingParams, ModelRegistry
from megrez_moe import MegrezMoeForCausalLM
from vllm.entrypoints.cli.main import main


ModelRegistry.register_model(
    "MegrezMoeForCausalLM",
    MegrezMoeForCausalLM,
)


if __name__ == "__main__":
    main()
