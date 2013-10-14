from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GL import *
import sys


class abstractScene:
	def __init__(self):
		self.w = 400
		self.h = 400
	def initialize(self):
		glClearColor(0.0,0.0,0.0,0.0)
	def update(self):pass
	def render(self):
		glClear(GL_COLOR_BUFFER_BIT)
		glLoadIdentity()
		glFlush()
	def resize(self, w, h):
		glViewport (0, 0, w, h)
		glMatrixMode (GL_PROJECTION)
		glLoadIdentity ()
		glFrustum (-1.0, 1.0, -1.0, 1.0, 1.5, 20.0)
		glMatrixMode (GL_MODELVIEW)
		print "Hello"

	def glutKeyPress(self, key, x, y):
		print "hello"
	def glutKeyRelease(self, key, x, y):
		print "there"
	def glutMouse(self, x, y):
		print x, y
	def glutMouseClick(self, button, state, x, y):
		print "hello"

class glapplication:
	def __init__(self, scene):
		self.scene = scene

		self.initializeScene()
		self.initializeCallbacks()

	def initializeScene(self):
		glutInit(sys.argv)
		glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH)
		glutInitWindowSize(self.scene.w, self.scene.h)
		glutCreateWindow("Connect 4")
		self.initializeCallbacks()
		glutMainLoop()

	def cleanupScene(self):pass
	def initializeCallbacks(self):
		glutDisplayFunc(self.scene.render)
		glutReshapeFunc(self.scene.resize)
		glutIdleFunc(self.scene.update)
		glutKeyboardFunc(self.scene.glutKeyPress)
		glutKeyboardUpFunc(self.scene.glutKeyRelease)
		glutMouseFunc(self.scene.glutMouseClick)
		glutPassiveMotionFunc(self.scene.glutMouse)


def main():
	scene = abstractScene()
	app = glapplication(scene)


if __name__ == '__main__':
	main()