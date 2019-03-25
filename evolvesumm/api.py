import evolvesumm


def process_args(args):
    pop = int(args.get('pop', 100))
    summ = float(args.get('summ', 0.1))
    gens = int(args.get('gens', 500))
    cross = float(args.get('cross', 0.5))
    scale = float(args.get('scale', 0.5))
    fit = args.get('fit')
    sim = args.get('sim')
    txt_in = args.get('txt_in', '')
    if txt_in:
        txt_in = txt_in.strip()

    try:
        txt_out = evolvesumm.evolesumm(txt_in, summary_length=summ, population_size=pop,
                                       iterations=gens, lambda_=scale, crossover_rate=cross,
                                       fitness=fit)
    except Exception as exc:
        txt_out = ''

    return pop, summ, gens, cross, scale, fit, sim, txt_in, txt_out
