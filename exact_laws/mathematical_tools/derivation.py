import numpy as np


def cdiff_2point(tab, length_case=1):
    if length_case == 0:
        return 0
    tab = np.array(tab)
    return (tab[-1] - tab[0]) / (2 * length_case)


def cdiff_4point(tab, length_case=1):
    if length_case == 0:
        return 0
    tab = np.array(tab)
    return (tab[0] + 8 * tab[-2] - 8 * tab[1] - tab[-1]) / (12 * length_case)


def cdiff_point(tab, length_case=1, precision=4):
    if precision == 4:
        return cdiff_4point(tab, length_case)
    elif precision == 2:
        return cdiff_2point(tab, length_case)


def cdiff_prec4(tab, length_case=1, dirr=0, period=True):
    if len(np.shape(tab)) == 1:
        dirr = 0
    if length_case == 0:
        return np.zeros(np.shape(tab))
    tab = np.array(tab)
    result = (
        np.roll(tab, 2, axis=int(dirr))
        + 8 * np.roll(tab, -1, axis=int(dirr))
        - 8 * np.roll(tab, 1, axis=int(dirr))
        - np.roll(tab, -2, axis=int(dirr))
    ) / (12 * length_case)
    if not period:
        if dirr == 0:
            result[0] = (tab[1] - tab[0]) / length_case
            result[1] = (tab[2] - tab[0]) / (2 * length_case)
            result[-2] = (tab[-1] - tab[-3]) / (2 * length_case)
            result[-1] = (tab[-1] - tab[-2]) / length_case
        elif dirr == 1:
            result[:, 0] = (tab[:, 1] - tab[:, 0]) / length_case
            result[:, 1] = (tab[:, 2] - tab[:, 0]) / (2 * length_case)
            result[:, -2] = (tab[:, -1] - tab[:, -3]) / (2 * length_case)
            result[:, -1] = (tab[:, -1] - tab[:, -2]) / length_case
        elif dirr == 2:
            result[:, :, 0] = (tab[:, :, 1] - tab[:, :, 0]) / length_case
            result[:, :, 1] = (tab[:, :, 2] - tab[:, :, 0]) / (2 * length_case)
            result[:, :, -2] = (tab[:, :, -1] - tab[:, :, -3]
                                ) / (2 * length_case)
            result[:, :, -1] = (tab[:, :, -1] - tab[:, :, -2]) / length_case
    return result


def cdiff_prec2(tab, length_case=1, dirr=0, period=True):
    if len(np.shape(tab)) == 1:
        dirr = 0
    if length_case == 0:
        return np.zeros(np.shape(tab))
    tab = np.array(tab)
    result = (np.roll(tab, -1, axis=int(dirr)) -
              np.roll(tab, 1, axis=int(dirr))) / (2 * length_case)
    if not period:
        if dirr == 0:
            result[0] = (tab[1] - tab[0]) / length_case
            result[-1] = (tab[-1] - tab[-2]) / length_case
        elif dirr == 1:
            result[:, 0] = (tab[:, 1] - tab[:, 0]) / length_case
            result[:, -1] = (tab[:, -1] - tab[:, -2]) / length_case
        elif dirr == 2:
            result[:, :, 0] = (tab[:, :, 1] - tab[:, :, 0]) / length_case
            result[:, :, -1] = (tab[:, :, -1] - tab[:, :, -2]) / length_case
    return result


def cdiff_array(tab, length_case=1, dirr=0, period=True, precision=4):
    if precision == 4:
        return cdiff_prec4(tab, length_case, dirr, period)
    elif precision == 2:
        return cdiff_prec2(tab, length_case, dirr, period)


def cdiff(tab, length_case=1, dirr=0, precision=4, period=True, point=False):
    """
    Return the differential of 'tab' according to a quantity whose length is 'length_case' whith a centre method \n
    Caution 1: if 'tab' has more than one dimension, don't forget to indicate the direction ('dirr') of the differential (default=0) \n
    Caution 2: periodical edge effects by default (period=True) \n
    Caution 3: 'length_case' is the length of a case \n
    précision ordre 2 (default if not period): need to know the one-step-left and right values \n
    précision ordre 4 (default): need to know the two-step-left and right values \n
    Remark : np.roll shift the table to the right (the end comes to the begining)
    """
    if point:
        return cdiff_point(tab, length_case, precision)
    else:
        return cdiff_array(tab, length_case, dirr, period, precision)


def div(tab_vec, case_vec=[None], precision=4, period=True):
    """
    Return the divergence of 'tab_vec' by a vectorial quantity whose resolution is 'case_vec' (cartesian coordinates)
    Caution 1: 'tab_vec' has to have the same number of dimensions than the 'case_vec' length
    Caution 2: 'case_vec' is the length of the cases in each directions
    for 'precision' and 'period', see 'cdiff' function description
    """
    if case_vec[0] is None:
        case_vec = np.ones(len(tab_vec), dtype=int)
    return np.sum(
        [cdiff(tab_vec[i], case_vec[i], i, precision, period)
         for i in range(len(case_vec))],
        axis=0,
    )


def rot_gen(tab_vec, case_vec=[None], precision=4, period=True):
    if case_vec[0] is None:
        case_vec = np.ones(len(tab_vec), dtype=int)
    for i in range(len(case_vec)):
        if i == 0:
            A = cdiff(tab_vec[2], case_vec[1], 1, precision, period)
            B = cdiff(tab_vec[1], case_vec[2], 2, precision, period)
        elif i == 1:
            A = cdiff(tab_vec[0], case_vec[2], 2, precision, period)
            B = cdiff(tab_vec[2], case_vec[0], 0, precision, period)
        elif i == 2:
            A = cdiff(tab_vec[1], case_vec[0], 0, precision, period)
            B = cdiff(tab_vec[0], case_vec[1], 1, precision, period)
        yield A - B


def rot(tab_vec, case_vec=[None], precision=4, period=True):
    """
    Return the rotational of 'tab_vec' by a vectorial quantity whose resolution is 'case_vec'
    Caution 1: 'tab_vec' has to have the same number of dimensions than the 'case_vec' length
    Caution 2: 'case_vec' is the length of the cases in each directions
    for 'precision' and 'period', see 'cdiff' function description
    """
    return list(rot_gen(tab_vec, case_vec, precision, period))


def grad_gen(tab, case_vec=[None], precision=4, period=True):
    if case_vec[0] is None:
        case_vec = np.ones(len(np.shape(tab)), dtype=int)
    for i in range(len(case_vec)):
        out = cdiff(tab, case_vec[i], i, precision, period)
        yield out


def grad(tab, case_vec=[None], precision=4, period=True):
    """
    Return the gradient of 'tab' by a vectorial quantity whose resolution is 'case_vec'
    Caution 1: 'tab' has to have the same number of dimensions than the 'case_vec' length
    Caution 2: 'case_vec' is the length of the cases in each directions
    for 'precision' and 'period', see 'cdiff' function description
    """
    return list(grad_gen(tab, case_vec, precision, period))


def laplacien2(tab, alpha=[None], case_vec=[None], precision=4, period=True):
    if case_vec[0] is None:
        case_vec = np.ones(len(np.shape(tab)), dtype=int)
    if alpha[0] is None:
        alpha = np.ones(len(np.shape(tab)), dtype=int)
    return np.sum(
        [alpha[i]*cdiff(cdiff(tab, case_vec[i], i, precision, period), case_vec[i], i, precision, period)
         for i in range(len(case_vec))],
        axis=0,
    )

def cdiff2_prec4(tab, length_case=1, dirr=0, period=True):
    if len(np.shape(tab)) == 1:
        dirr = 0
    if length_case == 0:
        return np.zeros(np.shape(tab))
    tab = np.array(tab)
    result = ( -30 * tab
        - np.roll(tab, 2, axis=int(dirr))
        + 16 * np.roll(tab, -1, axis=int(dirr))
        + 16 * np.roll(tab, 1, axis=int(dirr))
        - np.roll(tab, -2, axis=int(dirr))
    ) / (12 * length_case * length_case)

    return result

def laplacien(tab, alpha=[None], case_vec=[None]):
    if case_vec[0] is None:
        case_vec = np.ones(len(np.shape(tab)), dtype=int)
    if alpha[0] is None:
        alpha = np.ones(len(np.shape(tab)), dtype=int)
    return np.sum(
        [alpha[i]*cdiff2_prec4(tab, case_vec[i], i)
         for i in range(len(case_vec))],
        axis=0,
    )
