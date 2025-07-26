import aiohttp
import asyncio
import random

async def poll_for_voxel(session):
    while True:
        async with session.get("http://localhost:5555/emit") as resp:
            emit = await resp.json()
            if emit.get("voxel"):
                async with session.get("http://localhost:5555/env/get_voxel") as voxel_resp:
                    if voxel_resp.ok:
                        voxel_data = (await voxel_resp.json()).get("voxel_data")
                        print(f"Received voxel data: {voxel_data}")
                        return voxel_data
            await asyncio.sleep(0.5)

async def send_reward(session, reward):
    await session.post("http://localhost:5555/env/send_reward", json={"reward": reward})

async def main():
    async with aiohttp.ClientSession() as session:
        while True:
            voxel_data = await poll_for_voxel(session)
            reward = random.uniform(0, 1)
            print(f"Sending reward: {reward}")
            await send_reward(session, reward)

if __name__ == "__main__":
    asyncio.run(main())
