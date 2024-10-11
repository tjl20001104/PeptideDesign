idea:
    phase 1:VAE-BAYES
    1.use unlabeled peptide to train a WAE in order to gain an unbiased generalised WAE(for antimicrobial peptide)
    2.use labeled peptide sequence to train a classifier on latent space, namely use encoder to map labeled peptide sequences to a list of z(zs) on latent space. and use zs to train a classifier(maybe Naive Bayes,no specific idea)
    3.generate z from p(z) and use classifier to determine if it's toxic,etc. Clearly this can also be viewed as an pure classifier.
    4.for the generated peptides,we can use another nlp models to check again how good it is and pick the best like 20 peptides.
    5.maybe use molecular dynamics to simulate

#data gathering -- done

#data prepossessing -- done

#data to vector -- done

#data_loader -- done

#model -- done
