from pydantic_ai import Agent, RunContext, Tool
from app.config import settings
from app.agents_call.agents import AgentCall
from typing import Any



class Responder:

    def __init__(self, prompts):
        self.prompt = prompts

    async def generate(self, call, query, chat_history):

        async def drug_discovery(ctx: RunContext[str]):

            return await call('drug_discovery', ctx.deps)


        async def clinical_trail(ctx: RunContext[str]):

            return  await call('clinical_trail', ctx.deps)

        async def drug_interaction(ctx: RunContext[str]):

            return await call('drug_interaction', ctx.deps)

        coordinate_agent = Agent(
            settings.MODEL,
            deps_type = list[dict],
            output_type = str,
            tools=[
                Tool(drug_discovery, takes_ctx = True),
                Tool(clinical_trail, takes_ctx = True),
                Tool(drug_interaction, takes_ctx = True) ],
            system_prompt=self.prompt['coordinate_agent_prompt']
        )

        if len(chat_history) > 1:
            @coordinate_agent.system_prompt
            async def coordinate_agent_prompt(ctx: RunContext[list[dict]]) -> str:
                return str(ctx.deps)
        try:
            answer = await coordinate_agent.run(query, deps = chat_history)
        except Exception as e:
            return e

        return answer






    async def __call__(self, query, chat_history):

        call = AgentCall(self.prompt)

        return  await self.generate(call, query, chat_history)

