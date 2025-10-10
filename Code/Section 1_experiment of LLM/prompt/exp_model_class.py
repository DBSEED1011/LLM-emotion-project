from enum import Enum


class ExtendedModelType(Enum):
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT_o3 = 'o3-mini-2025-01-31'
    Deepseek_R1 = "deepseek-r1"
    Deepseek_v3 = "deepseek-v3"

    @property
    def value_for_tiktoken(self) -> str:
        return self.value if self is not ExtendedModelType.STUB else "gpt-3.5-turbo"

    @property
    def is_openai(self) -> bool:
        r"""Returns whether this type of models is an OpenAI-released model."""
        return self in {
            ExtendedModelType.GPT_o3,
            ExtendedModelType.GPT_3_5_TURBO_0125,
        }

    @property
    def is_open_source(self) -> bool:
        r"""Returns whether this type of models is open-source."""
        return self in {
            ExtendedModelType.Deepseek_R1,
            ExtendedModelType.Deepseek_v3,
        }

    @property
    def token_limit(self) -> int:
        r"""Returns the maximum token limit for a given model.
        Returns:
            int: The maximum token limit for the given model.
        """
        if self is ExtendedModelType.GPT_o3:
            return 16385
        elif self is ExtendedModelType.GPT_3_5_TURBO_0125:
            return 4096
        elif self is ExtendedModelType.Deepseek_R1:
            return 16384
        elif self is ExtendedModelType.Deepseek_v3:
            return 16384
        else:
            raise ValueError("Unknown model type")
