def format_time(seconds):
    days = int(seconds // (24 * 3600))
    hours = int((seconds % (24 * 3600)) // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if days > 0:
        time_str = f'{days}d {hours}h {minutes}m {seconds}s'
    elif hours > 0:
        time_str = f'{hours}h {minutes}m {seconds}s'
    elif minutes > 0:
        time_str = f'{minutes}m {seconds}s'
    else:
        time_str = f'{seconds}s'

    return time_str