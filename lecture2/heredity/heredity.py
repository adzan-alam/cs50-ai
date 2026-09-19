import csv
import itertools
import sys

PROBS = {
    # Unconditional probabilities for having gene
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        # Probability of trait given two copies of gene
        2: {True: 0.65, False: 0.35},
        # Probability of trait given one copy of gene
        1: {True: 0.56, False: 0.44},
        # Probability of trait given no gene
        0: {True: 0.01, False: 0.99},
    },
    # Mutation probability
    "mutation": 0.01,
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):
        # Check if current set of people violates known information
        fails_evidence = any(
            (
                people[person]["trait"] is not None
                and people[person]["trait"] != (person in have_trait)
            )
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
                "trait": (
                    True
                    if row["trait"] == "1"
                    else False
                    if row["trait"] == "0"
                    else None
                ),
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s)
        for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.
    """
    p = 1.0

    for person, data in people.items():
        # Determine how many genes this person has in this universe
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0

        # Determine whether this person exhibits the trait in this universe
        has_trait = person in have_trait

        # 1. Calculate the probability of the person having these genes
        mother = data["mother"]
        father = data["father"]

        if mother is None and father is None:
            # No parents listed: use unconditional probability
            gene_prob = PROBS["gene"][genes]
        else:
            # Parents listed: calculate inheritance probabilities
            def p_pass_gene(parent_name):
                """Helper to calculate probability that a parent passes on the gene."""
                if parent_name in two_genes:
                    # Parent passes gene unless it mutates
                    return 1 - PROBS["mutation"]
                elif parent_name in one_gene:
                    # Parent passes gene 50% of the time, subject to mutation
                    return 0.5
                else:
                    # Parent doesn't pass gene, unless it mutates
                    return PROBS["mutation"]

            p_mom = p_pass_gene(mother)
            p_dad = p_pass_gene(father)

            if genes == 2:
                gene_prob = p_mom * p_dad
            elif genes == 1:
                gene_prob = (p_mom * (1 - p_dad)) + ((1 - p_mom) * p_dad)
            else:  # genes == 0
                gene_prob = (1 - p_mom) * (1 - p_dad)

        # 2. Calculate the probability of having/not having the trait given these genes
        trait_prob = PROBS["trait"][genes][has_trait]

        # Combine into total probability for this person and multiply to joint total
        p *= gene_prob * trait_prob

    return p


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    """
    for person in probabilities:
        # Determine person's gene count in the current universe
        if person in two_genes:
            genes = 2
        elif person in one_gene:
            genes = 1
        else:
            genes = 0

        # Update gene probability sum
        probabilities[person]["gene"][genes] += p

        # Determine person's trait status in the current universe
        has_trait = person in have_trait

        # Update trait probability sum
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # Normalize "gene" distribution
        gene_sum = sum(probabilities[person]["gene"].values())
        if gene_sum > 0:
            for g in probabilities[person]["gene"]:
                probabilities[person]["gene"][g] /= gene_sum

        # Normalize "trait" distribution
        trait_sum = sum(probabilities[person]["trait"].values())
        if trait_sum > 0:
            for t in probabilities[person]["trait"]:
                probabilities[person]["trait"][t] /= trait_sum


if __name__ == "__main__":
    main()
