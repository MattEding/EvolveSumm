import flask

import api


app = flask.Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def index():
    args = flask.request.args
    pop, summ, gens, cross, scale, fit, sim, txt_in, txt_out = api.process_args(args)
    return flask.render_template('summarizer.html', pop=pop, gens=gens,
                                 cross=cross, scale=scale, fit=fit, sim=sim,
                                 txt_in=txt_in)


if __name__ == '__main__':
    app.run(debug=True)
