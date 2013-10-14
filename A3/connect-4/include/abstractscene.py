import sys

try:
	from OpenGL.GLUT import *
	from OpenGL.GL import *
	from OpenGL.GLU import *
except:
	print '''
ERROR: PyOpenGL not installed properly.
		'''


class abstractScene(object):
	def __init__(self):
		self.w = 500
		self.h = 500

	def initialize(self):
		abstract
	def update(self):
		abstract
	def render(self):
		abstract
	def resize(self, w, h):
		abstract
	def glutKeyPressEvent(self, key, x, y):
		abstract
	def glutKeyReleaseEvent(self, key, x, y):
		abstract
	def glutMousePassiveMoveEvent(self, x, y):
		abstract
	def glutMouseActiveMoveEvent(self, x, y):
		abstract
	def glutMouseClickEvent(self, button, state, x, y):
		abstract

class glApplication(object):
	def __init__(self, scene):
		self.scene = scene

		self.initializeScene()
		self.initializeCallbacks()
		glutMainLoop()

	def initializeScene(self):
		glutInit(sys.argv)
		glutInitDisplayMode(GLUT_SINGLE | GLUT_DEPTH | GLUT_RGBA)
		glutInitWindowSize(self.scene.w, self.scene.h)
		glutCreateWindow("Connect 4")
		self.scene.initialize()

	def cleanupScene(self):pass

	def initializeCallbacks(self):
		glutDisplayFunc(self.scene.render)
		glutReshapeFunc(self.scene.resize)
		glutIdleFunc(self.scene.update)
		glutKeyboardFunc(self.scene.glutKeyPressEvent)
		glutKeyboardUpFunc(self.scene.glutKeyReleaseEvent)
		glutMouseFunc(self.scene.glutMouseClickEvent)
		glutPassiveMotionFunc(self.scene.glutMousePassiveMoveEvent)
		glutMotionFunc(self.scene.glutMouseActiveMoveEvent)