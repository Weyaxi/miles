import asyncio

from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import GatewayConfig


class _Backend:
    def __init__(self):
        self.unloaded = []

    async def load_slot(self, slot, rank, alpha):
        pass

    async def unload_slot(self, slot):
        self.unloaded.append(slot)

    async def export_slot(self, *args, **kwargs):
        pass

    async def sample(self, *args):
        await asyncio.Event().wait()

    def trainer_dead(self):
        return False


def test_expiry_only_reclaims_the_owning_sessions_resources(tmp_path):
    async def run():
        backend = _Backend()
        service = TinkerService(backend, GatewayConfig(base_model="base", n_slots=2, checkpoint_root=str(tmp_path)))
        sessions = [service.create_session("tenant") for _ in range(2)]
        models = [
            service.create_model("tenant", {"session_id": session, "model_seq_id": 1, "base_model": "base"})[1]
            for session in sessions
        ]
        await asyncio.gather(*service._create_tasks)
        samplers = [
            service.create_sampling_session("tenant", {"session_id": session, "sampling_session_seq_id": 1})
            for session in sessions
        ]
        async with service._trainer_lock:
            exported = await service._save_weights_for_sampler(service.models[models[0]], {})
        implicit_sampler = exported["sampling_session_id"]
        assert service.sampling_sessions[implicit_sampler]["session_id"] == sessions[0]
        slots = [service.models[model].slot for model in models]
        pending_train = [service.submit("tenant", "optim_step", {"model_id": model, "seq_id": 1}) for model in models]
        pending_sample = [
            service.submit_sample("tenant", {"sampling_session_id": sampler, "seq_id": 1})[0] for sampler in samplers
        ]
        direct_sample = service.submit_sample("tenant", {})[0]
        tasks = [entry[0] for entry in service._sample_tasks.values()]
        service.sessions[sessions[0]]["last_heartbeat"] -= 2 * service.config.lease_timeout_s
        try:
            # No yield before sweeping: even a sample cancelled before starting must settle its future.
            await service._sweep_once()
            assert set(service.sessions) == {sessions[1]}
            assert set(service.models) == {models[1]}
            assert set(service.sampling_sessions) == {samplers[1]}
            assert backend.unloaded == [slots[0]]
            assert service.free_slots == {slots[0]}
            for request_id in (pending_train[0], pending_sample[0]):
                future = service.futures.get(request_id, "tenant")
                assert future.state == "failed" and future.error == "lease expired"
            for request_id in (pending_train[1], pending_sample[1], direct_sample):
                assert service.futures.get(request_id, "tenant").state == "pending"
            await service._sweep_once()
            assert backend.unloaded == [slots[0]]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_heartbeat_while_sweeper_waits_for_trainer_preserves_session(tmp_path):
    async def run():
        service = TinkerService(_Backend(), GatewayConfig(base_model="base", n_slots=1, checkpoint_root=str(tmp_path)))
        session = service.create_session("tenant")
        _, model = service.create_model("tenant", {"session_id": session, "model_seq_id": 1, "base_model": "base"})
        await asyncio.gather(*service._create_tasks)
        service.sessions[session]["last_heartbeat"] -= 2 * service.config.lease_timeout_s
        async with service._trainer_lock:
            sweep = asyncio.create_task(service._sweep_once())
            await asyncio.sleep(0)
            service.heartbeat("tenant", session)
        await sweep
        assert session in service.sessions
        assert model in service.models
        assert not service.backend.unloaded

    asyncio.run(run())
