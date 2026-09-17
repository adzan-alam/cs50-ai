import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print("PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # first make a set consisting of all the pages linked to current page
    links = corpus[page]  # tis is a set

    # make in new dict with key = pages in corpus , probality = their chance of being next page which is (1-d)/no of pages
    if len(links) == 0:
        tr_model = dict.fromkeys(corpus, (1 / len(corpus)))
        return tr_model

    random_prob = (1 - damping_factor) / len(corpus)
    tr_model = dict.fromkeys(corpus, random_prob)

    # for links in the pages , they will get additional probabilty other thand random onne, which is rp + (damping_factor)/total no of pages

    no_of_links = len(links)
    for link in links:
        tr_model[link] += damping_factor / no_of_links
    return tr_model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Intialize new dict,chose random page , update its value
    pagerank = dict.fromkeys(corpus, 0)

    current_page = random.choice(list(corpus))

    # call tr_model , chose random page by chosing given probablities, sut current page to chose page
    for i in range(n):
        pagerank[current_page] += 1
        tr_model = transition_model(corpus, current_page, damping_factor)
        page = random.choices(list(tr_model.keys()), list(tr_model.values()), k=1)
        current_page = page[0]

    # Now we normalize the count
    for page, count in pagerank.items():
        pagerank[page] = count / n

    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    """
    num_pages = len(corpus)

    # 1. Initialize all pages with equal PageRank (1 / N)
    ranks = {page: 1 / num_pages for page in corpus}

    # Pre-handle pages with no links: treat them as linking to ALL pages
    adjusted_corpus = {}
    for page, links in corpus.items():
        if len(links) == 0:
            adjusted_corpus[page] = set(corpus.keys())
        else:
            adjusted_corpus[page] = links

    # 2. Repeatedly update ranks until no value changes by > 0.001
    while True:
        new_ranks = {}
        max_change = 0

        for page in corpus:
            # Sum up PR(i) / NumLinks(i) for every page 'i' that links to 'page'
            link_sum = 0
            for possible_linker in corpus:
                if page in adjusted_corpus[possible_linker]:
                    link_sum += ranks[possible_linker] / len(
                        adjusted_corpus[possible_linker]
                    )

            # Apply the PageRank formula
            new_ranks[page] = ((1 - damping_factor) / num_pages) + (
                damping_factor * link_sum
            )

            # Track the maximum change across all pages in this iteration
            change = abs(new_ranks[page] - ranks[page])
            max_change = max(max_change, change)

        # Update the ranks dictionary with new values
        ranks = new_ranks

        # 3. Check for convergence threshold
        if max_change < 0.001:
            break

    return ranks


if __name__ == "__main__":
    main()
