
## DDT1-06 - Nucleotide analogue library (Main problem)
Development of nucleotide analogs library by performing virtual screening using molecular docking methodologies at the active site of RNA dependent RNA polymerase enzyme of SAS-COV-2

## DDT1-11 - Glycan docking (Main problem)
To design small molecules that can target the glycan shield of SARS-CoV-2 using any methodology. The sugar molecules (glycans) should be correctly modelled as per latest MassSpec data. Molecules binding these sugar moieties can be identified from (a) virtual screening (b) Literature search of molecules binding different sugar moieties (c) machine learning/AI approaches from a data set of molecules that bind sugar moieties. The strength of the binding must be shown by estimating binding free energy on the glycosylated residue of choice. The top 100 or top 25% compounds from your list should be validated by providing a binding site and binding free energy from free energy calculations.

## DDT2-04 - Generate molecules and dock (Main problem)
The sequence identity of the COVID19 protease and that of SARS-CoV is high, hence by using the known SARS-CoV protease drugs generate possible drugs using machine learning methods to generate novel drug like candidates. Use a variational autoencoder with SMILES representation to generate novel molecules from the trained continuous latent space.

docking, MD of the best molecules

## DDT2-10 - GANs for peptide (Main problem)
The challenge is to either build GAN’S for bioactive peptide generation from scratch using python based deep learning frameworks or customize existing GAN implementations developed for new molecule generation based on SMILES . Success criterion is a python pipeline that utilizes GAN’s to generate potential bioactive peptides < 2000 kDa.

## DDT2-14 - Liver Injury (Main problem)

## PS-2-04
Use multiple SARS-CoV-2 protease conformation to screen of generated chemicals using ensemble molecular docking protocol at both allosteric and orthosteric sites. Retain chemicals with best docking scores and binding energies as per the guidelines. Shortlist the top 100 or top 25% (max 100) hits and refine the best 10 using MD simulation without any restraints.

Assessment Criteria:
SMARTS: The molecules should not have unlikely structures
Tanimoto similarity coefficients: Should not be more than 60% similar to training set
Presumably they will also look at binding energies

Input Form: max 4 mb
Parameters
Smiles	

Docking Score 	

Free energy of binding 	
	
Model in Pickle format	

Generated chemicals in smiles - what does this mean?

Proposed Functional, technical design documents - guess this means the python scripts and explanation of how it works

Innovation & Differentiation - “Describe how you would enhance. differentiate and add novelty to your approach. This can be in data & feature preparation , modelling, software development, user interfaces or scientific assumptions”


Common Form
Docking Form
MD Form
