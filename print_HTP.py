def print_HTP(H, T, P, filename="output.txt"):
    """
    Print matrices to .txt-file (name of file = filename).
    H... overall assembled stiffness matrix
    T... nodal temperature vector
    P... nodal force vector

    Make sure, that your system of equations is sorted by
    ascending node numbers, i.e., N1 N2 ... N100.
    """

    F = open(filename, 'w')

    F.write("Stiffness matrix H: \n")
    for row in H:
        for col in row:
            outline = "{0:+8.4e},".format(col)
            F.write("{0:11s}".format(str(outline)))
        F.write("\n")

    F.write("Temperature T: \n")
    for row in T:
        for col in row:
            outline = "{0:+8.4e},".format(col)
            F.write("{0:11s} \n".format(str(outline)))


    F.write("Force vector P: \n")
    for row in P:
        for col in row:
            outline = "{0:+8.4e},".format(col)
            F.write("{0:11s} \n".format(str(outline)))

    F.close()

    return None
