import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle

start = [None, None]
end = [None, None]

#Get user selection from mouse positions
def line_select_callback(eclick, erelease):
	start[:] = eclick.xdata, eclick.ydata
	end[:] = erelease.xdata, erelease.ydata


#Close plot when button pressed
def toggle_selector(event):
	if event.key in ['enter', ' '] and toggle_selector.RS.active:
		print('Selection Made')
		plt.close()


#Show plot for user selections
def select_spaxel(data, rect=None, width=1, height=1, date='UNSPECIFIED'):
	#Set sensible colourmap range based off average values across image
	ave = np.mean(data)
	#Clip within an order of mag of average
	data = np.clip(data, ave/5, ave*5)

	#Show image
	fig, ax = plt.subplots()
	ax.imshow(data, origin='lower', extent=[0,25,0,38])
	ax.title.set_text('Date: {}'.format(date))
	#Add box showing saved section if section already saved
	if rect != None:
		print(rect)
		ax.add_patch(Rectangle(rect, width, height, alpha=0.5, color='limegreen'))
	
	# drawtype is 'box' or 'line' or 'none'
	toggle_selector.RS = RectangleSelector(ax, line_select_callback,
										   drawtype='box', useblit=True,
										   button=[1, 1],  # don't use middle button
										   minspanx=5, minspany=5,
										   spancoords='pixels',
										   interactive=True)
	plt.connect('key_press_event', toggle_selector)

	plt.show()

	#Save coordinates of selection
	x = {'start':int(np.rint(start[0])), 'end':int(np.rint(end[0]))}
	y = {'start':int(np.rint(start[1])), 'end':int(np.rint(end[1]))}

	plt.close()

	print('x: {}'.format(x))
	print('y: {}'.format(y))

	return x,y