def displacement_amplitude(lim, step_size):
    """
    This function creates the amplitudes of displacement for cyclic tests.
    :param lim: the largest amplitude.
    :param step_size: increment of displacement.
    :return: [a1, a1, a2, a2.... lim, lim] where an is the amplitude at load cycle n.
    """
    disp_amps = []
    amp_current = 0
    stop_check = 0
    while stop_check < lim:
        new_amp = amp_current + step_size
        if new_amp > lim:
            new_amp = lim
        disp_amps.append(new_amp)
        disp_amps.append(new_amp)
        stop_check = disp_amps[-1]
        amp_current = amp_current + step_size
    return disp_amps


if __name__ == '__main__':
    x = tuple([1.23124123, 23.24123123, 242.245123123])
    print("{:.2f}".format(x))
