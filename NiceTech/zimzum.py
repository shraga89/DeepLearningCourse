if __name__=="__main__":
    out = open("C:/NiceTech/final_data_set/out.csv",'w')
    line = 0
    with open("C:/NiceTech/final_data_set/data_set.csv") as data_set:
        for record in data_set:
            if line >=1000:
                break
            out.write(record)
            line+=1
    out.close()