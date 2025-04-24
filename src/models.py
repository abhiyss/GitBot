from enum import Enum


class EmbeddingModel(Enum):
    AMAZON_TITAN_V2 = "amazon.titan-embed-text-v2:0"
    AMAZON_TITAN_V1 = "amazon.titan-embed-text-v1:0"


class EmbeddingModelDisplayNames(Enum):
    AMAZON_TITAN_V2 = "Amazon Titan V2"
    AMAZON_TITAN_V1 = "Amazon Titan V1"


class LLMModel(Enum):
    CLAUDE_SONNET_3 = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_HAIKU_3 = "anthropic.claude-3-haiku-20240307-v1:0"
    MISTRAL_INSTRUCT_7B = "mistral.mistral-7b-instruct-v0:2"


class LLMModelDisplayNames(Enum):
    CLAUDE_SONNET_3 = "Anthropic Claude Sonnet 3"
    CLAUDE_HAIKU_3 = "Anthropic Claude Haiku 3"
    MISTRAL_INSTRUCT_7B = "Mistral 7B Instruct"
