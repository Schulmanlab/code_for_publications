from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage import io 
import sys
import numpy as np 
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy.spatial.distance import euclidean
from scipy.spatial import Delaunay, ConvexHull
import networkx as nx
import math
import os
import glob 
from scipy.interpolate import splprep, splev
from scipy import ndimage as ndi
from skimage.feature import canny
from skimage.morphology import closing, disk
from skimage import img_as_uint, img_as_bool, img_as_float
from skimage.draw import circle 
from skimage.measure import label, regionprops, find_contours
from scipy import ndimage
from matplotlib import cm 


def calc_distance(endpoint1, endpoint2):
	#simple distance calculation
	distance_squared = (endpoint1[0]-endpoint2[0]) * (endpoint1[0]-endpoint2[0]) + (endpoint1[1]-endpoint2[1]) * (endpoint1[1]-endpoint2[1])
	distance = sqrt(distance_squared)

	return distance


def report_binding_fraction(seed_coords, ab_coords, cutoff_distance = 5.0):
	#simple method for calculating the binding fraction given an array of seed coordinates and an array
	#of antibody coordinates. Cutoff distance matches SI
	attached_seed_coordinates = []
	for seed_coord in seed_coords:
		for ab_coord in ab_coords:
			distance = calc_distance(seed_coord, ab_coord)
			if distance <= cutoff_distance:
				attached_seed_coordinates.append(seed_coord)
				break
	if len(seed_coords) == 0: #catching the case where nothing is detected
		attached_seed_percentage = 0.0
	else:
		attached_seed_percentage = float(len(attached_seed_coordinates))/float(len(seed_coords)) #calculating the percentage attached

	return attached_seed_percentage


def process_ab_seed_image_pair(ab_filename, seed_filename, membrane_radius = 10):
	#main function. Takes separate ab and seed images and returns the experimental binding fraction
	#also returns the "null" binding fraction found using random sampling

	#reading ab image
	image_ab = io.imread(ab_filename)

	#using laplacian of gaussian blob detection on blob image
	ab_blobs = blob_log(image_ab, max_sigma=30, num_sigma=10, threshold=.005)

	ab_coords_list = []

	'''blobs_list = [ab_blobs]
	colors = ['yellow']
	titles = ['Laplacian of Gaussian']
	sequence = zip(blobs_list, colors, titles)
	fig, ax = plt.subplots(figsize=(3, 3))
	ax.set_aspect('equal')
	#ax = axes.ravel()'''



	#reading seed image
	image_seed = io.imread(seed_filename)

	#using laplacian of gaussian blob detection on seed image
	seed_blobs = blob_log(image_seed, max_sigma=30, num_sigma=10, threshold=.005)

	seed_coords_list = []



	#now we will perform edge detection and morphological filling on the seed image to find the perimeter of the cell
	
	#completely turn off dim pixels to reduce noise for the edge detection
	mask = image_seed < 10
	image_seed[mask] = 0

	#perform edge detection using the canny algorithm
	edges_open = canny(image_seed, 5, 2, 25) #these parameters seem to work well for sisi's images
	
	#performing closing algorithm to get a complete outline
	selem = disk(30)#originally 5
	edges = closing(edges_open, selem) 
	edges = img_as_uint(edges)
  
  	#now we will fill in the closed cell 
	fill_cell = ndi.binary_fill_holes(edges)

	#we only want to analyze the largest closed object in the image, this will be the cell of interest
	label_image = label(fill_cell)
	largest_area = 0.0
	for region in regionprops(label_image):
		area = region.area
		if area > largest_area:
			cell_region = region 
			largest_area = area 
		else:
			continue
	#now cell region should contain just the outlined cell

	#creating a blank image and copying over just the cell region
	img_cell = np.zeros((512,512), dtype=np.uint8)
	cell_coords_list = cell_region.coords.tolist()
	for coord in cell_coords_list:
		img_cell[coord[0], coord[1]] = 1

	#applying an edge detection kernel to find the edge pixels in img_cell 
	kernel = np.uint8([[1,  1, 1], [1, 10, 1], [1,  1, 1]])

	#now we convolve the kernel with the image 
	convolved_img_cell = ndimage.convolve(img_cell, kernel, mode='constant', cval = 1)

	#now produce an output mask where pixels with value 18 are turned off
	#pixels with value 18 represent pixels that were completely surrounded by other pixels, these are interior
	#pixels and not edge pixels
	edge_mask = np.zeros_like(convolved_img_cell)
	edge_mask[np.where(convolved_img_cell != 18)] = 1
	edge_mask[np.where(convolved_img_cell == 0)] = 0

	#we now have binary image where pixels with value 1 are edge pixels
	#gathering the indices of the edge pixels from the binary image
	row_indices, column_indices = np.nonzero(edge_mask)

	#removing indices at the border of the 512x512 image as these are not part of the cell membrane
	edge_indices = []
	for i in range(len(row_indices)):
		if row_indices[i] == 0 or row_indices[i] == 511:
			continue
		if column_indices[i] == 0 or column_indices[i] == 511:
			continue
		edge_indices.append([row_indices[i], column_indices[i]])

	#now for each edge pixel draw a circle of a cutoff radius
	#this should match the cutoff distance for considering an ab to be part of the membrane

	#creating a blank image to draw the membrane points onto
	img_membrane = np.zeros((512,512), dtype=np.uint8)

	#for each edge point in the membrane edge indices, we will draw a cirlcewith membrane radius
	#all these cirlces overlap to form the cell membrane image
	fig, ax = plt.subplots(figsize=(3, 3))
	for edge_point in edge_indices:
		rr, cc = circle(edge_point[0], edge_point[1], membrane_radius)

		try: 
			img_membrane[rr, cc] = 1
		except IndexError:
			continue

		c = plt.Circle((edge_point[1], edge_point[0]), membrane_radius, color='blue', linewidth=2, fill=True)
		ax.add_patch(c) 
	
	#plotting ab blobs
	for blob in ab_blobs:
	    y, x, r = blob
	    if r <= 2.0:
	    	c = plt.Circle((x, y), r*4, color='green', linewidth=0, fill=True)
	    	ax.add_patch(c)

	#plotting seed blobs
	for blob in seed_blobs:
	#print blob 
		y, x, r = blob
		if r <= 2.0:
			c = plt.Circle((x, y), r*8, color='red', linewidth=0, fill=True, alpha = .40)
			ax.add_patch(c)

	#saving processed image
	plt.savefig("plots/"+ab_filename+"processed_just_seeds.pdf")


	#culling the ab blobs to remove artifacts
	for ab in ab_blobs:
		if ab[2] <= 2.0:#only consider blobs with small radii, these are the seeds/abs
			ab_coordinates = ab[0:2]
			ab_coords_list.append(ab_coordinates)
	ab_coords_list = np.vstack(ab_coords_list) #converting to np array

	#culling the seed blobs to remove artifacts
	for seed in seed_blobs:
		if seed[2] <= 2.0:#only consider blobs with small radii, these are the seeds/abs
			seed_coordinates = seed[0:2]
			seed_coords_list.append(seed_coordinates)
	if len(seed_coords_list) == 0:
		return "no seeds", "no seeds" #if there are no seeds detected at all we need to exit
	seed_coords_array = np.vstack(seed_coords_list) #converting to array


	#further culling the ab points, if there aren't at least 3 other ab points nearby we can assume
	#that ab is not on the cell membrane and is either an artifact or an ab that has come off of the membrane
	culled_ab_points = []
	for point1 in ab_coords_list:
		n_neighbors = 0
		include_point = False
		for point2 in ab_coords_list:
			if point1[0] == point2[0] and point1[1] == point2[1]: #don't compare a point to itself
				continue
			p2p_distance = calc_distance(point1, point2)
			if p2p_distance <= 150: #150 pixels is a large distance so this should only be removing abs that are nowhere near the cell
				n_neighbors+=1
			if n_neighbors>=3:
				include_point = True 
				break

		if include_point == True:
			culled_ab_points.append(point1)
		else:
			print "excluding point!"

	culled_ab_points_array = np.vstack(culled_ab_points) #convertign to array

	#now we will only count abs that are within a cutoff distance of one of the edge pixels
	culled_ab_points_on_edge = []
	for point1 in culled_ab_points_array:
		include_point = False
		for point2 in edge_indices:
			if point1[0] == point2[0] and point1[1] == point2[1]: #if the point and the edge point overlap, count it
				culled_ab_points_on_edge.append(point1)
				continue
			p2p_distance = calc_distance(point1, point2)
			#print "distance is: ", p2p_distance
			if p2p_distance <= membrane_radius: #if they are within the membrane radius from an edge point, count it
				include_point = True 
				culled_ab_points_on_edge.append(point1)
				break

	#now we will only count seeds that are within a cutoff distance of one of the edge pixels
	membrane_radius = 10 
	culled_seeds = []
	for point1 in seed_coords_array:
		include_point = False
		for point2 in edge_indices:
			if point1[0] == point2[0] and point1[1] == point2[1]: #if the point and the edge point overlap, count it
				culled_seeds.append(point1)
				continue
			p2p_distance = calc_distance(point1, point2)
			#print "distance is: ", p2p_distance
			if p2p_distance <= membrane_radius: #if they are within the membrane radius from an edge point, count it
				include_point = True 
				culled_seeds.append(point1)
				break
	if len(culled_seeds) == 0: #if there are no seeds at this point, we need to exit
		return "no seeds", "no seeds"

	#converting to array
	seed_coords_array = np.vstack(culled_seeds)
	culled_ab_points_array_on_edge = np.vstack(culled_ab_points_on_edge)

	#determine experimental binding fraction from these culled lists
	culled_binding_fraction = report_binding_fraction(seed_coords_array, culled_ab_points_array_on_edge)

	#now for the random sampling part
	#the positive pixels in img_membrane represent the total "area" of the membrane
	#we want to randomly sample from these pixels
	row_indices_mem, column_indices_mem = np.where(img_membrane)
	mem_indices = np.column_stack((row_indices_mem,column_indices_mem)) #converting positive pixels into an array

	#now we will randomly pick a number of points in mem_indices equal to the number of seeds
	#and calculate the binding fraction
	#we will do this 1000 times and report the average null binding
	null_fractions = []
	for i in range(1000):
		rand_indices = np.random.randint(len(mem_indices), size=(len(seed_coords_array))) #the number of random membrane points to sample should match the number of detected seeds
		simulated_seed_coords_array = []
		for index in rand_indices:
			simulated_seed_coords_array.append(mem_indices[index])
		#now find the binding fraction using the simulated seed coords
		null_fraction = report_binding_fraction(simulated_seed_coords_array, culled_ab_points_array_on_edge)
		null_fractions.append(null_fraction)
	null_binding_fraction = sum(null_fractions)/len(null_fractions)


	return culled_binding_fraction, null_binding_fraction 
	

	


cell_dirs = ['1','2','3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
#time_dirs = [ '1', '2', '3', '4']
#time_dirs = ['1_test']



for cell_dir in cell_dirs:	
	ab_membrane_counts = []
	culled_binding_fractions = []
	null_binding_fractions = []
	ab_file_list = glob.glob(cell_dir+"/*AB*")
	seed_file_list = glob.glob(cell_dir+"/*S*")
	print ab_file_list
	print seed_file_list

	for i in range(len(ab_file_list)):
		ab_full_file = ab_file_list[i]
		seed_full_file = seed_file_list[i]
		print 'ab file is: ', ab_full_file
		print 'seed file is: ', seed_full_file
	
		culled_binding_fraction, null_binding_fraction = process_ab_seed_image_pair(ab_full_file, seed_full_file)
	
		if culled_binding_fraction == "no seeds":
			continue 

		culled_binding_fractions.append(culled_binding_fraction)
		null_binding_fractions.append(null_binding_fraction)
	
	if len(culled_binding_fractions) == 0:
		total_binding_fraction = "found 0"
		total_null_binding_fraction = "found 0 for total binding fraction"
	else:
		total_binding_fraction = sum(culled_binding_fractions)/float(len(culled_binding_fractions))
		total_null_binding_fraction = sum(null_binding_fractions)/float(len(null_binding_fractions))
	print "printing culled_binding_fractions"
	f1=open(cell_dir+'_culled_binding_fraction_optimal_membrane_test.dat','w+')
	print >>f1, total_binding_fraction
	f1.close()

	print "printing null_binding_fractions"
	f1=open(cell_dir+'_null_binding_fraction_optimal_membrane_test.dat','w+')
	print >>f1, total_null_binding_fraction
	f1.close()







