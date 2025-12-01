# Configure Graphiti
from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
from graphiti_core.driver.neo4j_driver import Neo4jDriver

from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

from graphiti_core.llm_client.config import LLMConfig

from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig

from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
import os
driver = Neo4jDriver(
    uri=os.environ.get('NEO4J_URI', 'neo4j://192.168.0.100:7687'),
    user=os.environ.get('NEO4J_USER', 'neo4j'),
    password=os.environ.get('NEO4J_PASSWORD', 'password'),
)
llm_config = LLMConfig(
    model = "gpt-4.1-mini",
    small_model="gpt-4.1-nano",
)

llm_client=OpenAIGenericClient(config=llm_config)
embedder=OpenAIEmbedder(
    config=OpenAIEmbedderConfig(
        model="text-embedding-3-small",
    )
)
# Pass the driver to Graphiti
llm_client
    