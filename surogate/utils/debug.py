def debug_breakpoint():
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=5678, stdout_to_server=True, stderr_to_server=True)