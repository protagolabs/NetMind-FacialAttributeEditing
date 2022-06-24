from flask import Flask
from dynaconf import settings
from flask_cors import CORS
from endpoints import demo

application = Flask(__name__)

application.config['SWAGGER_UI_DOC_EXPANSION'] = settings.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
application.config['RESTPLUS_VALIDATE'] = settings.RESTPLUS_VALIDATE
application.config['RESTPLUS_MASK_SWAGGER'] = settings.RESTPLUS_MASK_SWAGGER
application.config['ERROR_404_HELP'] = settings.RESTPLUS_ERROR_404_HELP

with application.app_context():
    application.register_blueprint(demo, url_prefix='/api')


@application.errorhandler(404)
def page_not_found(e):
    return 'error 404'


CORS(application)

if __name__ == "__main__":
    print('>>>>> Starting Deploy ProtagoLabs Demo server <<<<<')
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = False
    application.run(host='0.0.0.0', port=8080, threaded=False, processes=1, use_reloader=False)
