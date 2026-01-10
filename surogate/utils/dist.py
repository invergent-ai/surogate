def is_ray_worker() -> bool:
    """
    Check if code is running inside a Ray worker (actor).

    Returns:
        True if running in a Ray worker, False otherwise (running on driver/head node).
    """
    try:
        import ray
        if not ray.is_initialized():
            return False

        # Check if we're in a worker context
        ctx = ray.get_runtime_context()
        # Workers have actor_id set, driver doesn't
        return ctx.actor_id is not None
    except (ImportError, AttributeError, RuntimeError):
        # Ray not available, not in Ray context, or old Ray version
        return False


def is_ray_head() -> bool:
    """
    Check if code is running on the Ray head/driver node.

    Returns:
        True if running on head node (driver), False if in worker.
    """
    try:
        import ray
        if not ray.is_initialized():
            # Not in Ray context, assume head/driver
            return True

        # Check if we're in the driver context (not a worker)
        ctx = ray.get_runtime_context()
        # Driver has no actor_id
        return ctx.actor_id is None
    except (ImportError, AttributeError, RuntimeError):
        # Ray not available or not in Ray context, assume head
        return True


def get_ray_worker_rank() -> int:
    """
    Get the rank of the current Ray worker.

    Returns:
        Worker rank (0-indexed), or -1 if not in a distributed Ray context.
    """
    try:
        import ray
        if not ray.is_initialized() or not is_ray_worker():
            return -1

        # Try to get rank from runtime context
        ctx = ray.get_runtime_context()
        # This would need to be set by the application
        # For now, return -1 as rank needs to be tracked separately
        return -1
    except (ImportError, AttributeError, RuntimeError):
        return -1
