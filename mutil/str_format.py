
import datetime


def get_time_string(format="%Y%m%d_%H%M"):
    time_str = datetime.datetime.now().strftime(format)
    return time_str