from datetime import datetime
from os import rename
from pickle import dump, load

import os.path

timestamp = datetime.now().strftime("%Y-%m-%dT%H_%M_%S%z")
log_filename = "logfile_{}.pickle".format(timestamp)
logging = True

def write_log(key, data):
    if not logging:
        return
    # TODO: Keep file open?
    with open(log_filename, "ab") as log_file:
        dump(log_file, (key, data))

def load_log(log_file):
    items = []
    with open(log_file, "rb") as log_file:
        while True:
            try:
                items.append(load(log_file))
            except EOFError:
                break
    ret = dict()
    for key, data in items:
        if key in ret:
            entry = ret[key]
            if type(entry) == list:
                entry.append(data)
            else:
                ret[key] = [entry, data]
        else:
            ret[key] = data
    return ret

def change_logfile_name(new_filename):
    global log_filename
    if os.path.exists(log_filename):
        os.rename(log_filename, new_filename)
    log_filename = new_filename

def stop_logging():
    logging = False
def start_logging():
    logging = True
