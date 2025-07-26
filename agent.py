import aiohttp
import asyncio
import random

async def send_voxel(session, data):
    await session.post("http://localhost:5555/agent/send_voxel", json={"data": data})

async def poll_for_reward(session):
    while True:
        async with session.get("http://localhost:5555/emit") as resp:
            emit = await resp.json()
            if emit.get("reward"):
                async with session.get("http://localhost:5555/agent/get_reward") as reward_resp:
                    if reward_resp.ok:
                        reward = (await reward_resp.json()).get("reward")
                        print(f"Received reward: {reward}")
                        return
            await asyncio.sleep(0.5)

async def main():
    async with aiohttp.ClientSession() as session:
        while True:
            data = [random.choice([0, 1]) for _ in range(10)]
            print(f"Sending voxel data: {data}")
            await send_voxel(session, data)
            await poll_for_reward(session)
            await asyncio.sleep(1)  # Optional: wait before next round

if __name__ == "__main__":
    asyncio.run(main())
