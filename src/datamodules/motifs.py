"""Script handling motifs for evaluation"""

import pandas as pd

class Motif():
    """Motif class contains motif name, its sequence and its id used for indicating it
    """

    def __init__(self, name, seq, id, regex_str=None) -> None:

        # if we pass regex_string, use that instead of str for matching
        if regex_str is None:
            self.regex = seq
        else:
            self.regex = regex_str

        self.seq = seq
        self.name = name
        self.id = id
        self.where = None
        assert len(seq) != 0, "Pass a sequence of length>0"
 
    def __len__(self) -> int:
        # use sequence length
        return len(self.seq)

    def __str__(self):
        return self.name + " " + self.seq


    def ranges(self):
        # all indices
        motif_indices = self.where.nonzero()[0]
        motif_ranges = []
        i = 0

        #print(motif.where)
        #print(motif_indices)
        #print(self)
        #print(len(self))
        #print(motif_indices[-10:])
        while (i < len(motif_indices)):
            # if complete range we take it
            if motif_indices[i + len(self) - 1] - motif_indices[i] == len(self)-1:
                # looks goo, add the motif to list
                motif_ranges.append((int(motif_indices[i]),int(motif_indices[i]+len(self))))
                i += len(self)
            else: 
                print("what")
                print("The given size probably does not match the regex")
                # should not happen if non overlapping motif indications
                # this motif not complete
                i+=1
            
        return motif_ranges

class MotfiHandler():
    """Handles all motifs in df
    """

    def __init__(self, motifs) -> None:

        # convert all motif tuples to len 4
        motifs = [m if len(m)==4 else (m[0],m[1],m[2],None) for m in motifs]

        self.df = pd.DataFrame(motifs, columns=["name","seq","id", "regex_str"])
        self.dict = {k:v for (k,v) in zip(list(self.df.seq),list(self.df.id))}
        self.name_seq_dict = {k:v for (k,v) in zip(list(self.df.name),list(self.df.seq))}
        
        self.motifs = []
        # create motif objects
        for idx, row in self.df.iterrows():
            self.motifs.append(Motif(**dict(row)))

        self.df["motif"] = self.motifs

    def __len__(self) -> int:
        return len(self.df)
    
    def get_motif(self, seq : str =None,name : str=None,id : int=None) -> Motif:
        """Get motif by defining either sequence, name or its id. Only pass one of these.

        Args:
            seq (str, optional): motif sequence string. Defaults to None.
            name (str, optional): motif name. Defaults to None.
            id (int, optional): motif id used. Defaults to None.

        Returns:
            Motif: returns defined motif instance
        """
        if seq is not None:
            if seq=="non_motif":
                return "non-motif"
            return self.df[self.df["seq"]==seq].iloc[0]["motif"]
            #Motif(**dict(self.df[self.df["seq"]==seq].iloc[0]))
        if name is not None: 
            if name=="non_motif":
                return "non-motif"
            return self.df[self.df["name"]==name].iloc[0]["motif"]
            #Motif(**dict(self.df[self.df["name"]==name].iloc[0]))
        if id is not None:
            return self.df[self.df["id"]==id].iloc[0]["motif"]
            #Motif(**dict(self.df[self.df["id"]==id].iloc[0]))

    def __iter__(self):
        return iter(self.motifs)
    """
    def __iter__(self):
        self.cidx = 0
        return self

    def __next__(self):
        if self.cidx >= self.__len__():
            raise StopIteration
        motif = Motif(**dict(self.df.iloc[self.cidx]))
        self.cidx += 1
        return motif 
    """


sacc_cer_utr3_motifs = [("Puf3","TGTAAATA",1),
                        #("Puf3_all","TGTA*ATA",17,"TGTA[TACG]ATA"),
                        ("Puf3_TA","TGTA*ATA",19,"TGTA[TA]ATA"),
                        ("Puf3_T","TGTATATA",18),
                        ("Puf3_C","TGTACATA",17),
                        ("Puf3_G","TGTAGATA",17),
                        ("Whi3","TGCAT",2),
                        ("Rmd9l","ATATTC",3),
                        #("Pub1","TTTTTTA",4),
                        ("Puf2","TAATAAT",5),
                        ("Pab1","TATATA",6),
                        ("Pin4","TTTAATGA",7),
                        ("Nrd1","TTCTTGT",8)]

sacc_cer_utr3_motif_handler = MotfiHandler(sacc_cer_utr3_motifs)

pombe_utr3_motifs = [("Puf3","TGTAAATA",1),
                        #("Puf3_AT","TGTA*ATA",111,"TGTA[TA]ATA"),
                        #("Puf3_ATC","TGTAXATA",111,"TGTA[CTA]ATA"),
                        ("Pin4?","TTAATGA",11),
                        ("Pin4","TTTAATGA",101),
                        #("TATTTAT","TATTTAT",12),
                        ("ACTAAT","ACTAAT",13),
                        ("Are","TTATTTATT",14)]
pombe_utr3_motif_handler = MotfiHandler(pombe_utr3_motifs)


puf_motifs = [("Puf3","TGTAAATA",1),
                ("Puf3_T","TGTATATA",22),
                ("Puf3_C","TGTACATA",33),
                ("Puf3_G","TGTAGATA",34),
                ("Puf3_all","XXXXTGTA*ATAXXXX",41,"[TCGA]{4}TGTA[ACT]ATA[TCGA]{4}"),
                ("Puf3_l_A","XXXXTGTATATAXXXX",42,"[TCGA]{4}TGTAAATA[TCGA]{4}"),
                ("Puf3_l_T","XXXXTGTAAATAXXXX",43,"[TCGA]{4}TGTATATA[TCGA]{4}"),
                ("Puf3_l_C","XXXXTGTAAATAXXXX",44,"[TCGA]{2}C[TCGA]TGTA[ACT]ATA[TCGA]{4}"),
                ]
                #("Pub1","TTTTTTA",4),
                #("Puf4","TGTATAATA",40)]

puf_motif_handler = MotfiHandler(puf_motifs)


puf_motifs_c = [("Puf3_no_c","**TGTA*ATA",26,"[TGA][TGAC]TGTA[TA]ATA"),
                ("Puf3_c","C*TGTA*ATA",27,"C[TGAC]TGTA[TA]ATA")]

pufc_motif_handler = MotfiHandler(puf_motifs_c)

sacc_cer_utr3_motifs_flanks = [("Puf3","NNNNTGTA*ATANNNN",1,"[TCGA]{4}TGTA[TAC]ATA[TCGA]{4}"),
                        #("Puf3_all","TGTA*ATA",17,"TGTA[TACG]ATA"),
                        ("Whi3","NNNNTGCATNNNN",2,"[TCGA]{4}TGCAT[TCGA]{4}"),
                        ("Rmd9l","NNNNATATTCNNNN",3,"[TCGA]{4}ATATTC[TCGA]{4}"),
                        #("Pub1","TTTTTTA",4),
                        ("Puf2","NNNNTAATAATNNNN",5,"[TCGA]{4}TAATAAT[TCGA]{4}"),
                        ("Pab1","NNNNTATATANNNN",6,"[TCGA]{4}TATATA[TCGA]{4}"),
                        ("Pin4","NNNNTTTAATGANNNN",7,"[TCGA]{4}TTTAATGA[TCGA]{4}"),
                        ("Nrd1","NNNNTTCTTGTNNNN",8,"[TCGA]{4}TTCTTGT[TCGA]{4}")]


scer_motif_flanks_handler = MotfiHandler(sacc_cer_utr3_motifs_flanks)