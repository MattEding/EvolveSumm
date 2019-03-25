import evolvesumm


def process_args(args):
    pop = int(args.get('pop'))
    summ = int(args.get('summ'))
    gens = int(args.get('gens'))
    cross = float(args.get('cross'))
    scale = float(args.get('scale'))
    fit = args.get('fit')
    sim = args.get('sim')
    txt_in = args.get('txt_in')
    # txt_out = evolvesumm.evolvesumm(txt_in)

    print(list(map(type, [sim, fit, scale, cross, gens, summ, pop])))

    try:
        txt_out = evolvesumm.evolesumm(txt_in, summary_length=summ, population_size=pop, iterations=gens, lambda_=scale,
                      crossover_rate=cross, fitness='coh_sep')
    except Exception as exc:
        print('FAILED', exc)
        txt_out = None


    return pop, summ, gens, cross, scale, fit, sim, txt_in, txt_out
