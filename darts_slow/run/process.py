if __name__ == "__main__":
    for l in open("2_noise.txt", encoding="utf16"):
        if "Valid" in l and "Final Prec@1 " in l:
            print(l[48:].strip())
