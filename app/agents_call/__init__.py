from pydantic_ai import Agent, RunContext, Tool
from app.config import settings
from typing import Any



class Responder:

    def __init__(self, prompts):
        self.prompt = prompts
        self.drug_discovery_agent = Agent(settings.MODEL,system_prompt = self.prompt['drug_discovery_prompt'])
        self.clinical_trail_agent = Agent(settings.MODEL,system_prompt = self.prompt['clinical_trail_prompt'])
        self.drug_interaction_agent = Agent(settings.MODEL,system_prompt = self.prompt['drug_interaction_prompt'])

    async def generate(self, query, history):

        async def drug_discovery(ctx: RunContext[None]) -> str:
            try:
                r = await self.drug_discovery_agent.run(query, deps = ctx.deps)
                return str(r.output)
            except Exception as e:
                return e


        async def clinical_trail(ctx: RunContext[None]) -> str:

            try:
                r =  await self.clinical_trail_agent.run(query, deps = ctx.deps)
                return str(r.output)
            except Exception as e:
                return e

        async def drug_interaction(ctx: RunContext[None]) -> str:

            try:
                r =  await self.drug_interaction_agent.run(query, deps = ctx.deps)
                return str(r.output)
            except Exception as e:
                return e
        coordinate_agent = Agent(
            settings.MODEL,
            deps_type = list[dict],
            tools=[
                Tool(drug_discovery, takes_ctx = True),
                Tool(clinical_trail, takes_ctx = True),
                Tool(drug_interaction, takes_ctx = True) ],
            system_prompt=self.prompt['coordinate_agent_prompt']
        )

        if len(history) > 1:
            @coordinate_agent.system_prompt
            async def coordinate_agent_prompt(ctx: RunContext[list[dict]]) -> str:
                return str(ctx.deps)
        try:
            answer = await coordinate_agent.run(query, deps = history)
        except Exception as e:
            return e

        return answer






    async def __call__(self, query, chat_history):


        final_value = await self.generate(query, chat_history)

        return final_value

