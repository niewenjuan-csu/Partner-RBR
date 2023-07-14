fclose('all');
% path = 'F:\protein_rna\feature\train';
path = 'F:\protein_rna\feature\test';
protein = dir(path);
for i = 3: length(protein)
    eachprotein = protein(i).name;
    snbpssm(path, eachprotein); 
end
% protein = 'P01327';
% protein = 'A0K9S9';
% [A1]=snbpssm(path, protein);
% fclose('all')
    
    