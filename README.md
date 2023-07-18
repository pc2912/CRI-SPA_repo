# CRI-SPA_repo
Handle CRI-SPA data


### Notebooks <br/>
**Plate_Shuffling.ipynb** <br/>
Instruction to shuffle colonies when scaling from 384 to 1536 with a pinning robot. <br/>
**Read_Plates.ipynb** <br/>
Extract and process data from images. <br/>
**Screen Comparisons.ipynb** <br/>
Assess screens reproducibility. <br/>
**Networks.ipynb** <br/>
Go Enrichment Analysis in clusters of genes. <br/>
Draws networks of enriched terms. <br/>
**Liquid Confirmation.ipynb** <br/>
Plot the fluorescence of hit strains grown in liquid SC.<br/>


### Supporting Functions <br/>
Colony_Analysis.py, Plate_Shuffling.py <br/>

### Data Files <br/>
**QD_1.csv:** (Quadruplicate) Size and Yellowness raw and corrected data for colony quadruplicates. Data extracted from pictures in screen_sample/Screen1_9.9.21/ with Read_Plates.ipynb <br/>
**GA_1.csv:** (Gene Analysis). Mean and std of colonies for each gene, raw and corrected data, data extracted from pictures in screen_sample/Screen1_9.9.21/ with Read_Plates.ipynb  <br/>
**Screens Descriptions::** <br/>
**2.csv:**  Screen repeat with final antibiotic selection <br/>
**4.csv:**  Screen repeat with final antibiotic selection <br/>
**6.csv:**  Screen repeat with final antibiotic selection <br/>
**7.csv:**  Screen repeat without final antibiotic selection <br/>
**1_2_4_6.csv:** data extracted from several screen repeats, pooled and processed after pooling. <br/>
**For each screen, both QD and GA are available ** <br/>

### Raw Data <br/>
**screen_sample/Screen1_9.9.21/** Images (24H) for screen1<br/>
**BTX_chess/H48.png** Image used to compare BY_Ref and BY_Btx in Fig 3A<br/>
**Liquid_Data/** Fluorescent Data for Hit Strains in liquid SC <br/>
**Liquid_Data/BTX_stability.23.4.10** Fluorescent Data Confirming BTX stability for a week (Fig. S7) <br/>

**gene2go, go.obo:** For Go Enrichment Analysis. <br/>
**Plate_Map.txt:** Stores gene positions in our KO library. <br/>

