import csv
import itertools
import sys

import numpy as np

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])
    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    parents = []
    children = []
    for human in people.keys():
        if people[human]['mother'] is None:
            parents.append(human)
        else:
            children.append(human)

    parents_joint = 1

    for parent in parents:
        n_genes = 1 * (parent in one_gene) + 2 * (parent in two_genes)
        trait = (parent in have_trait)

        parents_joint *= PROBS['gene'][n_genes] * PROBS['trait'][n_genes][trait]

    children_joint = 1
    for child in children:
        mother = people[child]['mother']
        father = people[child]['father']

        n_genes = 1 * (child in one_gene) + 2 * (child in two_genes)
        n_genes_mother = 1 * (mother in one_gene) + 2 * (mother in two_genes)
        n_genes_father = 1 * (father in one_gene) + 2 * (father in two_genes)
        trait = (child in have_trait)
        mother_effect = PROBS['mutation'] * (n_genes_mother == 0) + 0.5 * (n_genes_mother == 1) + (1 - PROBS['mutation']) * (n_genes_mother == 2)
        father_effect = PROBS['mutation'] * (n_genes_father == 0) + 0.5 * (n_genes_father == 1) + (1 - PROBS['mutation']) * (n_genes_father == 2)
        if n_genes == 0:
            children_joint *= (1 - mother_effect) * (1 - father_effect)
        elif n_genes == 1:
            children_joint *= (1 - mother_effect) * father_effect + mother_effect * (1 - father_effect)
        else:
            children_joint *= mother_effect * father_effect

        children_joint *= PROBS['trait'][n_genes][trait]
    total_joint = parents_joint * children_joint

    return total_joint


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for human in probabilities:
        n_genes = 1 * (human in one_gene) + 2 * (human in two_genes)
        trait = (human in have_trait)
        probabilities[human]['gene'][n_genes] += p
        probabilities[human]['trait'][trait] += p

    return


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for human in probabilities:
        sum_genes_probs = 0
        for i in range(3):
            sum_genes_probs += probabilities[human]['gene'][i]

        for i in range(3):
            probabilities[human]['gene'][i] /= sum_genes_probs

        sum_trait_probs = probabilities[human]['trait'][True] + probabilities[human]['trait'][False]
        probabilities[human]['trait'][True] /= sum_trait_probs
        probabilities[human]['trait'][False] /= sum_trait_probs

    return


if __name__ == "__main__":
    main()
