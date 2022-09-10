def to_cvhsv(standard):
    return ((standard[0]/360) * 170, (standard[1]/100) * 255, (standard[2]/100) * 255)


def to_standard_hsv(cvhsv):
    return ((cvhsv[0]/170) * 360, (cvhsv[1]/255) * 100, (cvhsv[2]/255) * 100)
