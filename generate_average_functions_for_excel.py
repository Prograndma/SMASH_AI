def main(leap, column, how_many, starting):
    # =AVERAGE(K2:K49)
    starting -= 1
    string = ""
    for i in range(how_many):
        string += f"=AVERAGE({column}{(starting + (i * leap)) + 1}:{column}{starting + ((i + 1) * leap)})\n"

    print(string)


if __name__ == "__main__":
    main(24, "E", 66, 3)
