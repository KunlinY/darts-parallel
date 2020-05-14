if __name__ == "__main__":
    for l in open("1_noise.txt", encoding="utf8"):
        if "Valid" in l and "Final Prec@1 " in l:
            print(l[48:].strip())
