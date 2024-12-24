import time
import pynvml

def _read_sysfs_file(path):
    with open(path, "r") as f:
        return f.read().strip()

def get_cpu_reading():
    #return int(_read_sysfs_file("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"))
    files = [
        int(_read_sysfs_file("/sys/class/powercap/intel-rapl:0/energy_uj")),
        int(_read_sysfs_file("/sys/class/powercap/intel-rapl:1/energy_uj")),
    ]
    return sum(files)


# def eval_pass(inference, word, stats, handle, dev="gpu"):
#     stats.__reset__()


#     if dev == "gpu":
#         before_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
#     else:
#         before_energy = get_cpu_reading()

#     before_time = time.time()

#     try:
#         prediction = inference(word)
#     except Exception as e:
#         print(e)
#         #Misaligned pronoun here and ValueError
#         return 0, 0, None

#     after_time = time.time()
#     if dev == "gpu":
#         after_energy = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
#     else:
#         after_energy = get_cpu_reading()

#     time_delta   = after_time - before_time
#     energy_delta = after_energy - before_energy

#     return energy_delta, time_delta, prediction