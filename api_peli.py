from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
from des_peli import predict_movie_genre  # Importar la función de predicción

app = Flask(__name__)
CORS(app)

api = Api(
    app,
    version='1.0',
    title='Predicción de Géneros de Películas',
    description='API para predecir géneros de películas basados en la trama.'
)

ns = api.namespace('prediccion_genero', description='Predicción de géneros de películas')

resultado_modelo = api.model('Resultado', {
    'Titulo': fields.String,
    'Ano': fields.Integer,
    'Generos': fields.List(fields.String),
})

@ns.route('/')
class MovieGenrePredictionApi(Resource):
    @api.expect(resultado_modelo)
    @api.marshal_with(resultado_modelo)
    def get(self):
        titulo = request.args.get('titulo')
        ano = request.args.get('ano')

        try:
            generos = predict_movie_genre(titulo, ano)
            return {'Titulo': titulo, 'Ano': ano, 'Generos': generos}, 200
        except Exception as e:
            return {'error': str(e)}, 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
