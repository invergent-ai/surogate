import json

from ultra import docker_janitor
from ultra.docker_janitor import cleanup_stale_docker_networks, stale_compose_network_candidate


class Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _network(name, *, project=None, containers=None):
    labels = {}
    if project:
        labels["com.docker.compose.project"] = project
    return {
        "Id": f"{name}-id",
        "Name": name,
        "Labels": labels,
        "Containers": containers or {},
    }


def test_stale_compose_network_candidate_is_scoped_and_container_safe():
    stale = _network(
        "code_contests-0373__abc_default",
        project="code_contests-0373__abc",
    )
    active = _network(
        "code_contests-0373__active_default",
        project="code_contests-0373__active",
        containers={"container-id": {"Name": "runner"}},
    )
    unrelated = _network("other-project_default", project="other-project")

    assert stale_compose_network_candidate(stale) == (True, "stale")
    assert stale_compose_network_candidate(active) == (False, "active")
    assert stale_compose_network_candidate(unrelated) == (False, "unmatched")
    assert stale_compose_network_candidate(unrelated, all_compose=True) == (True, "stale")
    assert stale_compose_network_candidate(_network("bridge", project="bridge")) == (False, "protected")


def test_cleanup_stale_docker_networks_removes_only_detached_matching_networks(monkeypatch):
    networks = {
        "n1": _network("code_contests-0373__abc_default", project="code_contests-0373__abc"),
        "n2": _network(
            "code_contests-0373__active_default",
            project="code_contests-0373__active",
            containers={"container-id": {"Name": "runner"}},
        ),
        "n3": _network("other-project_default", project="other-project"),
    }
    removed = []

    def fake_available(docker_bin):
        return True

    def fake_run(args, *, docker_bin):
        if args[:2] == ["network", "ls"]:
            rows = "\n".join(json.dumps({"ID": network_id, "Name": data["Name"]}) for network_id, data in networks.items())
            return Proc(stdout=rows)
        if args[:2] == ["network", "inspect"]:
            return Proc(stdout=json.dumps([networks[args[2]]]))
        if args[:2] == ["network", "rm"]:
            removed.append(args[2])
            return Proc(stdout=args[2])
        raise AssertionError(args)

    monkeypatch.setattr(docker_janitor, "_docker_available", fake_available)
    monkeypatch.setattr(docker_janitor, "_run_docker", fake_run)

    dry = cleanup_stale_docker_networks(dry_run=True)

    assert dry["stale_candidates"] == 1
    assert dry["removed"] == 0
    assert dry["skipped"] == {"active": 1, "unmatched": 1}
    assert removed == []

    live = cleanup_stale_docker_networks(dry_run=False)

    assert live["removed"] == 1
    assert removed == ["code_contests-0373__abc_default-id"]
