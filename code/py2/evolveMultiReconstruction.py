from PyGMO import *
from pickle import *
from multiReconstructionProblem import *
import numpy as np
import VisualOdometry as VO

def evolveMultiReconstruction(filename = 'test', n_views=2, n_points=100, IPs=[], t_scale = 3.0, world_scale=10.0, K = [], roll=[], yaw=[], pitch=[], vx=[], vy=[], vz=[], snapshot_time_interval=0.25):
	""" Evolves a solution for a multiple view bundle adjustment problem (seeded by the available Astro Drone data).
	"""
	
	sd = 1234;
	prob = multiReconstructionProblem(n_views, n_points, IPs, t_scale, world_scale, K);
	n_genes = prob.getGenomeSize();
	print 'n_genes = %d' % n_genes
	
	# all values are scaled afterward:
	prob.lb = [-1.0] * n_genes;
	prob.ub = [1.0] * n_genes;

	# evolutions, generations and individuals
	n_evolutions = 10;
	n_generations = [10, 1, 10];#[1];#
	n_individuals = [50, 1, 50];#[1];#

	if(len(roll) == 0):
		seeded = False;
	else:
		seeded = True;
	
	if(not(seeded)):
		x = [];
		# prior on camera matrices and points:
		for v in range(n_views-1):
			x = x + [0.0] * 6;
		for p in range(n_points):
			x = x + [0.0, 0.5, 0.0];
	else:
		#print 'Prior genome: length = %d, max = %f, min = %f' % (len(x), np.max(np.array(x)), np.min(np.array(x)));
		(MRot, MTransl, MRotations, MTranslations) = VO.convertFromDroneToCamera(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval);
		X = VO.estimateWorldPoints(MRotations, MTranslations, IPs, K);
		x = constructMultiGenome(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval, X);
		fit_prior = prob._objfun_impl(x);
		print 'Prior fitness: %f' % fit_prior[0]; 
		

	print 'Creating algorithms'
	algo_list = []
	#algo_list.append(algorithm.py_cmaes(gen = n_generations[0]))
	algo_list.append(algorithm.pso_gen(gen = n_generations[0], vcoeff = 0.2));
	algo_list.append(algorithm.scipy_tnc(maxfun=150));
	algo_list.append(algorithm.sga(gen = n_generations[2], cr = 0.05, m = 0.05))
	
	# algo_list.append(algorithm.monte_carlo(iter = n_generations))
	# algo_list.append(algorithm.de(gen = n_generations))
	#algo_list.append(algorithm.py_cmaes(gen = n_generations[0]))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, vcoeff = 0.2));
	#algo_list.append(algorithm.pso_gen(gen = n_generations, vcoeff = 0.1));
	# algo_list.append(algorithm.pso_gen(gen = n_generations, vcoeff = 0.2));
	# algo_list.append(algorithm.pso_gen(gen = n_generations, vcoeff = 0.2));
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=3))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=4))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=5))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=6))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=5, neighb_type=1))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=5, neighb_type=3))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=5, neighb_type=4))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=6, neighb_type=1))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=6, neighb_type=3))
	#algo_list.append(algorithm.pso_gen(gen = n_generations, variant=6, neighb_type=4))

	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0, m = 0.05))
	#algo_list.append(algorithm.sga(gen = n_generations[2], cr = 0.05, m = 0.05))
	# algo_list.append(algorithm.sga(gen = n_generations, cr = 0.05, m = 0.01))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.5, m = 0.5))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.005, m = 0.001))
	
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03, selection = algorithm._algorithm._selection_type.BEST20))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03, selection = algorithm._algorithm._selection_type.BEST20))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03, selection = algorithm._algorithm._selection_type.BEST20))
	
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03, selection = algorithm._algorithm._selection_type.BEST20))
	#algo_list.append(algorithm.sga(gen = n_generations, cr = 0.1, m = 0.03, selection = algorithm._algorithm._selection_type.BEST20))

	topo = topology.ring()

	print 'Creating archipelago'
	archi = archipelago(topology = topo)
	# len(algo_list)
	
	n_islands = len(algo_list);

	for i in range(0, n_islands):
			print 'Island %d' % i
			if(not(seeded)):
				archi.push_back(island(algo_list[i], prob, n_individuals[i]))
			else:
				print 'starting with prior!'
				
				pop = population(prob);
				for ind in range(n_individuals[i]):
				
					if(ind >= 1):
						# perturb the individual:
						x = getPerturbedIndividual(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval, IPs, K);
					
					# push back the possibly perturbed individual:
					pop.push_back(x);
				archi.push_back(island(algo_list[i], pop))
			print 'Done with push back'

	avg_fits = np.zeros([n_islands,1]);
	
	for evs in range(0,n_evolutions):
		print '****************'
		print 'Evolving step %d' % evs
		print '****************'
		archi.evolve(1);
		archi.join()
		
		# save results until now
		# show the results for this step:
		i = 0
		isl_num = 1
		best_fit = 1E8;
		best_ind = 0;
		for isl in archi:
			f = open('genome_'+filename+'_'+str(isl_num), "a")
			f2 = open('fitness'+filename+'_'+str(isl_num), "a")	
			champ = isl.population.champion;
			isl_num = isl_num + 1
			print 'Isl. %d' % i,
			print 'fit %3.3f ' % champ.f,
			if(champ.f < best_fit):
				best_fit = champ.f;
				best_ind = i;
			if(evs == 0):
				avg_fits[i] = champ.f;
			else:
				avg_fits[i] = avg_fits[i] * 0.9 + champ.f[0] * 0.1;
			print 'avg_fit %3.3f' % avg_fits[i],
			
			# only numbers with spaces so that we can easily import them automatically afterwards:
			#f.write('x = array([')
			j = 1
			for ss in champ.x:
				f.write("".join(str(ss)))
				if j < len(champ.x):
					f.write(' ') #(',')
				j = j + 1
			#f.write('])')
			f.write('\n')
			f2.write("".join(str(champ.f)))
			f2.write('\n')
			i += 1
		
			f.close()
			f2.close()
		print ' '

	# return the best individual:
	genome = archi[best_ind].population.champion.x;
	(Rs, Ts, X) = prob.transformMultiGenomeToMatrices(genome, n_views, n_points);
	return (Rs, Ts, X);

def constructMultiGenome(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval, X):
	""" Constructs the genome.
	"""
	
	n_views = len(roll);
	n_world_points = len(X);
	world_scale = 10.0;
	t_scale = 3.0;
	angle_normalizer = 180.0; 
	
	genome = [];
	for v in range(n_views):
		genome += [roll[v] / angle_normalizer, yaw[v] / angle_normalizer, pitch[v] / angle_normalizer, ((vx[v]/ 1000.0)*snapshot_time_interval) / t_scale, ((vy[v]/ 1000.0)*snapshot_time_interval) / t_scale, ((vz[v]/ 1000.0)*snapshot_time_interval) / t_scale];

	for p in range(n_world_points):
		genome += (X[p,:] / world_scale).tolist();

	for g in range(len(genome)):
		gene = genome[g];
		if(gene >= 1.0):
			genome[g] = 0.9999;
		elif(gene <= -1.0):
			genome[g] = -0.9999;
	
	return genome;
	
def getPerturbedIndividual(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval, IPs, K):
	""" Perturbs the state estimates in order to get at different genomes
	"""
	
	# perturbation measurements:
	n_frames = len(roll);
	stv_angle = 0.5;
	stv_vel = 25.0;

	for fr in range(n_frames):
		noise = stv_angle * np.random.randn();
		roll[fr] += noise;
		noise = stv_angle * np.random.randn();
		yaw[fr] += noise;
		noise = stv_angle * np.random.randn();
		pitch[fr] += noise;
		noise = stv_vel * np.random.randn();
		vx[fr] += noise;
		noise = stv_vel * np.random.randn();
		vy[fr] += noise;
		noise = stv_vel * np.random.randn();
		vz[fr] += noise;
		
	# conversion to matrices:
	(MRot, MTransl, MRotations, MTranslations) = VO.convertFromDroneToCamera(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval);
	
	# estimation of world points:
	X = VO.estimateWorldPoints(MRotations, MTranslations, IPs, K);
	
	# construction of genome:
	x = constructMultiGenome(roll, yaw, pitch, vx, vy, vz, snapshot_time_interval, X);
	
	n_genes = len(x);
	for g in range(n_genes):
		if(type(x[g]) == complex):
			print 'Gene %d = complex' % (g)
			x[g] = np.abs(x[g]);
	
	return x;