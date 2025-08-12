from pydantic_ai import Agent
from app.config import settings



class AgentCall:


    def __init__(self, prompt):
        self.drug_discovery_agent = Agent(settings.MODEL,deps_type = str, output_type= str , system_prompt = prompt['drug_discovery_prompt'])
        self.clinical_trail_agent = Agent(settings.MODEL, deps_type = str,output_type = str, system_prompt = prompt['clinical_trail_prompt'])
        self.drug_interaction_agent = Agent(settings.MODEL, deps_type = str, output_type = str, system_prompt = prompt['drug_interaction_prompt'])

    async def __call__(self, value, context):

        match value:
            case "drug_discovery":
                r = await self.drug_discovery_agent.run(deps = context, usage = context.usage)
                return r.output
            case 'clinical_trail':
                r = await self.clinical_trail_agent.run(deps = context.deps, usage = context.usage)
                return r.output
            case 'drug_interaction':
                r = await self.drug_interaction_agent.run(deps = context.deps, usage = context.usage)
                return r.output


