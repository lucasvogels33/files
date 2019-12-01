
```python
#import necessary libraries

import csv
import sys
import numpy as np
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from random import *
import networkx as nx
from gurobipy import *
from timeit import default_timer as timer
from plotly import tools
import math
import copy

import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
```

```python
###########################################################################
##########################Create classes##################################
##########################################################################

class City:
  def __init__(self, name, numberofstreets, roadtype_vec,graph,piechart):
    self.name = name
    self.numberofstreets = numberofstreets
    self.roadtype_vec = roadtype_vec
    self.graph = graph
    self.piechart = piechart


#Create contractors with characteristics
class Contr:
  def __init__(self,color,TPA,PPA,k_factor):
    self.color = color
    self.TPA = TPA
    self.PPA = PPA
    self.k_factor = k_factor
    
class Iteration:
  def __init__(self,iterationNumber,TPA_precision,PPA_precision,AmountOfEdges):
      self.iterationNumber = iterationNumber
      self.TPA_precision = TPA_precision
      self.PPA_precision = PPA_precision
      self.AmountOfEdges = AmountOfEdges

class Component:
    def __init__(self,number,edges,size,contractor,TPAvector,PPAvector,adjacentComponents,adjacentContractors,swapvalue_regular,swapvalue_restricted,assigned):
      self.number = number
      self.edges = edges
      self.size = size
      self.contractor = contractor
      self.TPAvector = TPAvector
      self.PPAvector = PPAvector
      self.adjacentComponents = adjacentComponents
      self.adjacentContractors = adjacentContractors
      self.swapvalue_regular = swapvalue_regular
      self.swapvalue_restricted = swapvalue_restricted
      self.assigned = assigned
```

```python
##################################################################
############################define functions#####################
#################################################################
#function to plot current graph
def plotgraph(text):
    edge_trace= ()
    

    for k in range(NumberOfContractors):
        edge_trace = edge_trace +(go.Scatter(
            x=[],
            y=[],
            text=[],
            line=dict(width=1,color = Contractor[k].color),
            hoverinfo='none',
            mode='lines',
            ),)
            
    pre_edge_trace_x = []
    pre_edge_trace_y = []
    for k in range(NumberOfContractors):
        pre_edge_trace_x.append([])
        pre_edge_trace_y.append([])
    
    for e in G.edges():
        x0 = G.nodes[G.edges[(e[0],e[1],0)]['start']]['x']
        y0 = G.nodes[G.edges[(e[0],e[1],0)]['start']]['y']
        x1 = G.nodes[G.edges[(e[0],e[1],0)]['end']]['x']
        y1 = G.nodes[G.edges[(e[0],e[1],0)]['end']]['y']
        for k in range(NumberOfContractors):
            if G.edges[(e[0],e[1],0)]['assignvector'][k] == 1:
                pre_edge_trace_x[k] += [x0,x1,None]
                pre_edge_trace_y[k] += [y0,y1,None]

    

    for k in range(NumberOfContractors):
        edge_trace[k]['x'] = tuple(pre_edge_trace_x[k])
        edge_trace[k]['y'] = tuple(pre_edge_trace_y[k])

    timevectortestplot.append(end-start)
    
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            color=[],
            size=0.1,
            line=dict(width=2)),
        textfont=dict(
            family='sans serif',
            size=10,
            color='#1f77b4'
        ))
              
    pre_node_trace_x =[]
    pre_node_trace_y =[] 
    pre_node_trace_text = []
    for n in G.nodes():
        x = G.nodes[n]['x']
        y = G.nodes[n]['y']
        pre_node_trace_x += [x]
        pre_node_trace_y += [y]  
        if text ==1:    
            pre_node_trace_text += [n]
    
    
    node_trace['x'] += tuple(pre_node_trace_x)
    node_trace['y'] += tuple(pre_node_trace_y)  
    if text == 1:
        node_trace['text'] += tuple(pre_node_trace_text)
#    node_trace['text'] += tuple([n])
 

    fig = go.Figure(data=edge_trace +(node_trace,),
                 layout=go.Layout(
                    title='<br>Network graph made with Python',
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    plot(fig, filename="networkx.html")
    
#function to update all values based on contractor assignment
def updatevalues():

    # set TPA, PPA to zero for all contractors
    for k in range(NumberOfContractors):
        Contractor[k].TPA = 0    
        Contractor[k].PPA = 0
        Contractor[k].edges = []
         
    #calculate new TPA and PPA values for each contractor
    for e in edgelist:
        for multiplicity in range(G.number_of_edges(e[0],e[1])):
            for k in range(NumberOfContractors):
                if  G.edges[(e[0],e[1],multiplicity)]['assignvector'][k] ==1:
                    Contractor[k].TPA +=  G.edges[(e[0],e[1],multiplicity)]['TPAvector'][k] 
                    Contractor[k].PPA +=  G.edges[(e[0],e[1],multiplicity)]['PPAvector'][k] 
                    Contractor[k].edges.append((e[0],e[1],multiplicity))
                    G.edges[(e[0],e[1],multiplicity)]['contractor'] = k

    
    # update adjacent contractors and border value for each node
    for n in G.nodes():
        G.nodes[n]['contractors'] = np.zeros(NumberOfContractors)
        bordernodes = list(G[n])
        for n1 in bordernodes:
            for m in range(G.number_of_edges(n,n1)):
                edge = (n,n1,m)
                k = G.edges[edge]['contractor']
                G.nodes[n]['contractors'][k] += 1  

        G.nodes[n]['border']= np.count_nonzero(G.nodes[n]['contractors'])

    #update totalTPA
    start = timer()
    TPA = 0
    kmax = -1
    for k in range(NumberOfContractors):
        if Contractor[k].TPA > TPA:
            kmax = k
            TPA = Contractor[k].TPA
    globals()['TPA'] = TPA 
    globals()['k_TPA_high']= kmax       

    #update totalPPA
    PPA = 1000000
    kmin = -1
    for k in range(NumberOfContractors):
        if Contractor[k].PPA < PPA:
            kmin = k
            PPA = Contractor[k].PPA
    globals()['PPA'] = PPA  
    globals()['k_PPA_low'] = kmin
     
    #update niceness measure
    Niceness = 0
    for n in G.nodes():
        Niceness += G.nodes[n]['border']
    globals()['niceness'] = Niceness
    
       
#function running algorithm optimizing TPA (PPA and border are not taken into account)
def TPAspreadfunc():
    try:
        timeModel = Model('timeOpt')
        totaltime = timeModel.addVar(vtype=GRB.CONTINUOUS,obj=1.0,name="maxTime")   
        
        #define all binary x_{e,k} variable
        global assignVariables
        assignVariables = {}
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                assignVariables[(e[0],e[1],m)] = []
                for k in range(NumberOfContractors):
                    assignVariables[(e[0],e[1],m)].append(timeModel.addVar(vtype=GRB.BINARY,name="assign[(%d,%d,%d),%d]" %(e[0],e[1],m,k)))
        
        #capacity constraints x_{e,k} variables        
        for k in range(NumberOfContractors):
            timeModel.addConstr(sum(sum(assignVariables[(e[0],e[1],m)][k]*G.edges[(e[0],e[1],m)]['TPAvector'][k] for m in range(G.number_of_edges(e[0],e[1]))) for e in edgelist) <= totaltime, 'Time[%d]'%k)

#       constraint assigning every edge to exactly one contractor
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                timeModel.addConstr(sum(assignVariables[(e[0],e[1],m)][k] for k in range(NumberOfContractors)) == 1, 'assigned[%d,%d,%d]'% (e[0],e[1],m))
        
        #optimize the model
        timeModel.Params.MIPgap = 0.01
        timeModel.setParam( 'OutputFlag', False )
        timeModel.optimize()
        
        #update the assignvector for each edge
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                for k in range(NumberOfContractors):
                    G.edges[(e[0],e[1],m)]['assignvector'][k] = assignVariables[(e[0],e[1],m)][k].x
        
        
    except GurobiError as err:
        print('Error code ' + str(err.errno) + ": " + str(err))
    
    except AttributeError:
        print('Encountered an attribute error')
    #

#function running algorithm optimizing PPA (TPA and border are not taken into account)
def PPAspreadfunc():
    try:
        profitModel = Model('profitOpt')
        minprofit = profitModel.addVar(vtype=GRB.CONTINUOUS,obj=1.0,name="minProfit")   
        
        global assignVariables
        assignVariables = {}
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                assignVariables[(e[0],e[1],m)] = []
                for k in range(NumberOfContractors):
                    assignVariables[(e[0],e[1],m)].append(profitModel.addVar(vtype=GRB.BINARY,name="assign[(%d,%d,%d),%d]" %(e[0],e[1],m,k)))
        
        #capacity constraints x_{e,k} variables        
        for k in range(NumberOfContractors):
            profitModel.addConstr(sum(sum(assignVariables[(e[0],e[1],m)][k]*G.edges[(e[0],e[1],m)]['PPAvector'][k] for m in range(G.number_of_edges(e[0],e[1]))) for e in edgelist) >= minprofit, 'Profit[%d]'%k)

        #constraint assigning every edge to exactly one contractor
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                profitModel.addConstr(sum(assignVariables[(e[0],e[1],m)][k] for k in range(NumberOfContractors)) == 1, 'assigned[%d,%d,%d]'% (e[0],e[1],m))
        
        #optimize model
        profitModel.Params.MIPgap = 0.01
        profitModel.setParam( 'OutputFlag', False )
        profitModel.modelSense = GRB.MAXIMIZE
        profitModel.optimize()
        
        #update the assignvector for each edge
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                for k in range(NumberOfContractors):
                    G.edges[(e[0],e[1],m)]['assignvector'][k] = assignVariables[(e[0],e[1],m)][k].x
            
        
    except GurobiError as err:
        print('Error code ' + str(err.errno) + ": " + str(err))
    
    except AttributeError:
        print('Encountered an attribute error')
        
#function optimizing TPA while bounding PPA
def boundPPA_TPA_opt(PPAbound):
    try:
        time_profit_Model = Model('profit_time_Opt')
        maxtime = time_profit_Model.addVar(vtype=GRB.CONTINUOUS,obj=1.0,name="maxTime")   
        
        global assignVariables
        assignVariables = {}
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                assignVariables[(e[0],e[1],m)] = []
                for k in range(NumberOfContractors):
                    assignVariables[(e[0],e[1],m)].append(time_profit_Model.addVar(vtype=GRB.BINARY,name="assign[(%d,%d,%d),%d]" %(e[0],e[1],m,k)))
        
        #capacity constraints x_{e,k} variables        
        for k in range(NumberOfContractors):
            time_profit_Model.addConstr(sum(sum(assignVariables[(e[0],e[1],m)][k]*G.edges[(e[0],e[1],m)]['PPAvector'][k] for m in range(G.number_of_edges(e[0],e[1]))) for e in edgelist) >= PPAbound, 'Profit[%d]'%k)
            time_profit_Model.addConstr(sum(sum(assignVariables[(e[0],e[1],m)][k]*G.edges[(e[0],e[1],m)]['TPAvector'][k] for m in range(G.number_of_edges(e[0],e[1]))) for e in edgelist) <= maxtime, 'Profit[%d]'%k)    

        #constraint assigning every edge to exactly one contractor
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                time_profit_Model.addConstr(sum(assignVariables[(e[0],e[1],m)][k] for k in range(NumberOfContractors)) == 1, 'assigned[%d,%d,%d]'% (e[0],e[1],m))
        
        #optimize model
        time_profit_Model.Params.MIPgap = 0.01
        time_profit_Model.setParam( 'OutputFlag', False )
        time_profit_Model.modelSense = GRB.MINIMIZE
        time_profit_Model.optimize()
        
        #update the assignvector for each edge
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                for k in range(NumberOfContractors):
                    G.edges[(e[0],e[1],m)]['assignvector'][k] = assignVariables[(e[0],e[1],m)][k].x    
        
    except GurobiError as err:
        print('Error code ' + str(err.errno) + ": " + str(err))
    
    except AttributeError:
        print('Encountered an attribute error')



#function optimizng PPA while bounding TPA
def boundTPA_PPA_opt(TPAbound):
    try:
        profit_time_Model = Model('profit_time_Opt')
        minprofit = profit_time_Model.addVar(vtype=GRB.CONTINUOUS,obj=1.0,name="minProfit")   
        
        global assignVariables
        assignVariables = {}
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                assignVariables[(e[0],e[1],m)] = []
                for k in range(NumberOfContractors):
                    assignVariables[(e[0],e[1],m)].append(profit_time_Model.addVar(vtype=GRB.BINARY,name="assign[(%d,%d,%d),%d]" %(e[0],e[1],m,k)))
        
        #capacity constraints x_{e,k} variables        
        for k in range(NumberOfContractors):
            profit_time_Model.addConstr(sum(sum(assignVariables[(e[0],e[1],m)][k]*G.edges[(e[0],e[1],m)]['PPAvector'][k] for m in range(G.number_of_edges(e[0],e[1]))) for e in edgelist) >= minprofit, 'Profit[%d]'%k)
            profit_time_Model.addConstr(sum(sum(assignVariables[(e[0],e[1],m)][k]*G.edges[(e[0],e[1],m)]['TPAvector'][k] for m in range(G.number_of_edges(e[0],e[1]))) for e in edgelist) <= TPAbound, 'Profit[%d]'%k)    

        #constraint assigning every edge to exactly one contractor
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                profit_time_Model.addConstr(sum(assignVariables[(e[0],e[1],m)][k] for k in range(NumberOfContractors)) == 1, 'assigned[%d,%d,%d]'% (e[0],e[1],m))
        
        #optimize model
        profit_time_Model.Params.MIPgap = 0.01
        profit_time_Model.setParam( 'OutputFlag', False )
        profit_time_Model.modelSense = GRB.MAXIMIZE
        profit_time_Model.optimize()
        
        #update the assignvector for each edge
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                for k in range(NumberOfContractors):
                    G.edges[(e[0],e[1],m)]['assignvector'][k] = assignVariables[(e[0],e[1],m)][k].x    
        
    except GurobiError as err:
        print('Error code ' + str(err.errno) + ": " + str(err))
    
    except AttributeError:
        print('Encountered an attribute error')

#function updating all swapvalues used in the P_MRAP optimization
def update_swapvalues():
    
    #we change the order of the edgelist in this fucntion, we do not want to change the global variable edgelist
    edgelist_local = edgelist[:]
    
    #update regular swapvalues for each edge    
    for e in edgelist_local:
        for m in range(G.number_of_edges(e[0],e[1])): 
            edge = (e[0],e[1],m)
            G.edges[edge]['regular'] = np.zeros(NumberOfContractors) 
            kOld = G.edges[edge]['contractor']
            #calculate the swapvalue for normal edges
            for k in range(NumberOfContractors):
                if k == kOld:
                    G.edges[edge]['regular'][k] = 0
                else: 
                    swapvalue = 0
                    for x in [e[0],e[1]]:
                        NumberOfkOld = G.nodes[x]['contractors'][kOld]
                        NumberOfkNew = G.nodes[x]['contractors'][k]
                        if NumberOfkOld >= 2 and NumberOfkNew == 0:
                            swapvalue += 1
                        if NumberOfkOld == 1 and NumberOfkNew >= 1:    
                            swapvalue -= 1
                    G.edges[edge]['regular'][k] = swapvalue
    
    #update restricted swapvalues for each edge    
    for e in edgelist_local:
        for m in range(G.number_of_edges(e[0],e[1])): 
            edge = (e[0],e[1],m)
            G.edges[edge]['restricted'] = 0
    
    
    if randomrestrict == 1:
        shuffle(edgelist_local)
        
    for swapMin in [-2,-1,0]:
        for e in edgelist_local:
            for m in range(G.number_of_edges(e[0],e[1])): 
                edge = (e[0],e[1],m)
                if min(G.edges[edge]['regular']) == swapMin and G.edges[edge]['restricted'] == 0:
                    #keep current swapvalue
                    G.edges[edge]['restrictedswapvalue'] = G.edges[edge]['regular']
                    G.edges[edge]['restricted'] = 1
                    #restrict all adjacent not-yet-restricted edges from being swapped
                    for edge_adjacent in G.edges[edge]['adjacent_edges']:
                        if G.edges[edge_adjacent]['restricted'] == 0:
                                G.edges[edge_adjacent]['restricted'] = 1
                                assignedContractor = G.edges[edge_adjacent]['contractor']
                                G.edges[edge_adjacent]['restrictedswapvalue'] = [infty]*NumberOfContractors
                                G.edges[edge_adjacent]['restrictedswapvalue'][assignedContractor] = 0

#function updating all swapvalues of components used in the P_MRAP optimization
def update_components():  
    global Component_vec
    Component_vec = []
    
    for e in edgelist:
        for m in range(G.number_of_edges(e[0],e[1])): 
            edge = (e[0],e[1],m)
            G.edges[edge]['component'] = -1
    
    Current_component = -1
    for e in edgelist:
        for m in range(G.number_of_edges(e[0],e[1])): 
            edge = (e[0],e[1],m)
            if G.edges[edge]['component'] == -1:
                
                #add component
                Current_component +=1
                number = Current_component
                edges = [edge]
                size = 1
                comp_contractor = G.edges[edge]['contractor']
                TPAvector = G.edges[edge]['TPAvector']
                PPAvector = G.edges[edge]['PPAvector']
                adjacentComponents = []
                adjacentContractors = []
                swapvalue_regular = []
                swapvalue_restricted = []
                assigned = 0
                Component_vec.append(Component(number,edges,size,comp_contractor,TPAvector,PPAvector,adjacentComponents,adjacentContractors,swapvalue_regular,swapvalue_restricted,assigned))
                G.edges[edge]['component'] = Component_vec[Current_component]
                
                #add all adjacent edges when they have the same contractor
                oldedges = [edge]
                edgesadded = 1
                while edgesadded > 0: 
                    newedges = []
                    for edge_old in oldedges:
                        for edge_new in G.edges[edge_old]['adjacent_edges']:
                            if G.edges[edge_new]['contractor'] == comp_contractor and G.edges[edge_new]['component'] ==-1:
                                G.edges[edge_new]['component'] = Component_vec[Current_component]
                                Component_vec[Current_component].edges.append(edge_new)
                                Component_vec[Current_component].size += 1
                                Component_vec[Current_component].TPAvector = np.array(Component_vec[Current_component].TPAvector) + np.array(G.edges[edge_new]['TPAvector'])
                                Component_vec[Current_component].PPAvector = np.array(Component_vec[Current_component].PPAvector) + np.array(G.edges[edge_new]['PPAvector'])
                                newedges.append(edge_new)
                    edgesadded = len(newedges)
                    oldedges = newedges[:]
    
    #determine neigbouring components for each component
    for c in Component_vec:
        adjacentComponent_vec = []
        adjacentContractor_vec = []
        for edge in c.edges:
            for edge_adjacent in G.edges[edge]['adjacent_edges']:
                if G.edges[edge_adjacent]['contractor'] != c.contractor: 
                    adjComp = G.edges[edge_adjacent]['component']
                    adjcontr = G.edges[edge_adjacent]['contractor']
                    adjacentComponent_vec.append(adjComp)
                    adjacentContractor_vec.append(adjcontr)
        c.adjacentComponents = list(dict.fromkeys(adjacentComponent_vec))[:]
        c.adjacentContractors = list(dict.fromkeys(adjacentContractor_vec))[:]
    
    #determine regular swapvalues
    for c in Component_vec:
        swapvalue = [infty]*NumberOfContractors
        #assign swapvalue to component, forcing it to switch
        for k in range(NumberOfContractors):
            if k in c.adjacentContractors:
                swapvalue[k] = -1
            if k == c.contractor:
                swapvalue[k] = 0
        c.swapvalue_regular = swapvalue
        
    #determine restricted swapvalues for each component (step 0: sort all components based on their size)
    Component_vec = sorted(Component_vec, key=lambda x: x.size, reverse=False)
    
    #determine restricted swapvalues for each component (step 1: reset all components to unassigned)
    for c in Component_vec:
        c.assigned == 0
    #determine restricted swapvalues for each component (step 2: assign restricted swapvalues)   
    for c in Component_vec:
        if c.assigned == 0:
            swapvalue = [infty]*NumberOfContractors
            c.assigned = 1
            #assign swapvalue to component, forcing it to switch
            for k in range(NumberOfContractors):
                if k in c.adjacentContractors:
                    swapvalue[k] = -1
                if k == c.contractor:
                    swapvalue[k] = 0
            c.swapvalue_restricted = swapvalue
            #assign swapvalue to adjacent components, preventing it from moving
            for c_adj in c.adjacentComponents:
                swapvalue2 = [infty]*NumberOfContractors
                c_adj.assigned = 1
                for k in range(NumberOfContractors):
                    if k == c_adj.contractor:
                        swapvalue2[k] = 0
                c_adj.swapvalue_restricted = swapvalue2
            
                    
#function running the GAP algorithm on components    
def GAPalgorithm_comp(TPAbound,PPAbound,swapmethod):
    try:           
        global m_comp
        m_comp = Model("gapTest_comp")                
                
        global assignVariables_comp
        assignVariables_comp = {}
        for c in Component_vec:
            assignVariables_comp[c] = []
            for k in range(NumberOfContractors):
                if swapmethod == 'regular':
                    assignVariables_comp[c].append(m_comp.addVar(vtype=GRB.BINARY,obj=c.swapvalue_regular[k],name="assign[(%d),%d]" %(c.number,k)))
                if swapmethod =='restricted':
                    assignVariables_comp[c].append(m_comp.addVar(vtype=GRB.BINARY,obj=c.swapvalue_restricted[k],name="assign[(%d),%d]" %(c.number,k)))
        
        #capacity constraints x_{e,k} variables        
        for k in range(NumberOfContractors):
            m_comp.addConstr(sum(assignVariables_comp[c][k]*c.PPAvector[k] for c in Component_vec) >= PPAbound, 'Profit[%d]'%k)
            m_comp.addConstr(sum(assignVariables_comp[c][k]*c.TPAvector[k] for c in Component_vec) <= TPAbound, 'Time[%d]'%k)

        #constraint assigning every edge to exactly one contractor
        for c in Component_vec:        
            m_comp.addConstr(sum(assignVariables_comp[c][k] for k in range(NumberOfContractors)) == 1, 'assigned[%d]'%c.number)
        
        #warm start
        for c in Component_vec:
            for k in range(NumberOfContractors):
                if c.contractor == k:
                    assignVariables_comp[c][k].start = 1
                else:
                    assignVariables_comp[c][k].start = 0
    
        #optimize model
        m_comp.Params.MIPgap = 0.01
        m.Params.TimeLimit = 200
        m_comp.setParam( 'OutputFlag', False )
        m_comp.modelSense = GRB.MINIMIZE
        m_comp.optimize()
        
        #update the assignvector for each edge
        for c in Component_vec:
            for edge in c.edges:
                for k in range(NumberOfContractors):
                    G.edges[edge]['assignvector'][k] = assignVariables_comp[c][k].x    
        
    except GurobiError as err:
        print('Error code ' + str(err.errno) + ": " + str(err))
    
    except AttributeError:
        print('Encountered an attribute error')

#function running the GAP algorithm on the edges
def GAPalgorithm(TPAbound,PPAbound,swapmethod):
    try:           
        global m
        m = Model("gapTest")                
                
        global assignVariables
        assignVariables = {}
        for e in edgelist:
            for multi in range(G.number_of_edges(e[0],e[1])):
                assignVariables[(e[0],e[1],multi)] = []
                for k in range(NumberOfContractors):
                    assignVariables[(e[0],e[1],multi)].append(m.addVar(vtype=GRB.BINARY,obj=G.edges[(e[0],e[1],multi)][swapmethod][k],name="assign[(%d,%d,%d),%d]" %(e[0],e[1],multi,k)))
        
        #capacity constraints x_{e,k} variables        
        for k in range(NumberOfContractors):
            m.addConstr(sum(sum(assignVariables[(e[0],e[1],multi)][k]*G.edges[(e[0],e[1],multi)]['PPAvector'][k] for multi in range(G.number_of_edges(e[0],e[1]))) for e in edgelist) >= PPAbound, 'Profit[%d]'%k)
            m.addConstr(sum(sum(assignVariables[(e[0],e[1],multi)][k]*G.edges[(e[0],e[1],multi)]['TPAvector'][k] for multi in range(G.number_of_edges(e[0],e[1]))) for e in edgelist) <= TPAbound, 'Profit[%d]'%k)    

        #constraint assigning every edge to exactly one contractor
        for e in edgelist:
            for multi in range(G.number_of_edges(e[0],e[1])):
                m.addConstr(sum(assignVariables[(e[0],e[1],multi)][k] for k in range(NumberOfContractors)) == 1, 'assigned[%d,%d,%d]'% (e[0],e[1],multi))
        
        #warm start
        for e in edgelist:
            for multi in range(G.number_of_edges(e[0],e[1])):
                edge = (e[0],e[1],multi)
                for k in range(NumberOfContractors):
                    assignVariables[edge][k].start = G.edges[edge]['assignvector'][k]
            
        
        #optimize model
        m.Params.MIPgap = 0.01
        m.Params.TimeLimit = 200
        m.setParam( 'OutputFlag', False )
        m.modelSense = GRB.MINIMIZE
        m.optimize()
        
        #update the assignvector for each edge
        for e in edgelist:
            for multi in range(G.number_of_edges(e[0],e[1])):
                for k in range(NumberOfContractors):
                    G.edges[(e[0],e[1],multi)]['assignvector'][k] = assignVariables[(e[0],e[1],multi)][k].x    
        
    except GurobiError as err:
        print('Error code ' + str(err.errno) + ": " + str(err))
    
    except AttributeError:
        print('Encountered an attribute error')

        
#GUROBI method improving niceness directly
def optimize_border(TPAbound,PPAbound):
    try:
        global borderModel
        borderModel = Model('borderOpt')
        sumofspan = borderModel.addVar(vtype=GRB.CONTINUOUS,obj=1.0,name="minBorder")   
        global nodeVariables        
        global assignVariables
        assignVariables = {}
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                assignVariables[(e[0],e[1],m)] = []
                for k in range(NumberOfContractors):
                    assignVariables[(e[0],e[1],m)].append(borderModel.addVar(vtype=GRB.BINARY,name="assign[(%d,%d,%d),%d]" %(e[0],e[1],m,k)))
        
        #define all q_{n,k} variables
        nodeVariables = {}
        for n in G.nodes():
            nodeVariables[n] = []
            for k in range(NumberOfContractors):
                nodeVariables[n].append(borderModel.addVar(vtype=GRB.BINARY,name="border[%d,%d]" %(n,k)))
        
        #TPA and PPA constraints
        for k in range(NumberOfContractors):
            borderModel.addConstr(sum(sum(assignVariables[(e[0],e[1],m)][k]*G.edges[(e[0],e[1],m)]['PPAvector'][k] for m in range(G.number_of_edges(e[0],e[1]))) for e in edgelist) >= PPAbound, 'Profit[%d]'%k)
            borderModel.addConstr(sum(sum(assignVariables[(e[0],e[1],m)][k]*G.edges[(e[0],e[1],m)]['TPAvector'][k] for m in range(G.number_of_edges(e[0],e[1]))) for e in edgelist) <= TPAbound, 'Time[%d]'%k)
            
        #constraint assigning every edge to exactly one contractor
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                borderModel.addConstr(sum(assignVariables[(e[0],e[1],m)][k] for k in range(NumberOfContractors)) == 1, 'assigned[%d,%d,%d]'% (e[0],e[1],m))
        
        #constraint setting border objective value to sum of span
        for n in G.nodes():
            bordernodes = list(G[n])
            for n1 in bordernodes:
                for m in range(G.number_of_edges(n,n1)):
                    #edge only contained in one order in the assignvariables matrix
                    edge = (n,n1,m)
                    if edge not in assignVariables:
                        edge = (n1,n,m)
                    #add constraint    
                    for k in range(NumberOfContractors):
                        borderModel.addConstr(nodeVariables[n][k] >= assignVariables[edge][k],'bordering[%d,%d,%d,%d]'%(n,n1,m,k))
        
        borderModel.addConstr(sum(sum(nodeVariables[n][k] for k in range(NumberOfContractors)) for n in G.nodes()) == sumofspan, 'Niceness')
        
        #optimize model
        borderModel.Params.timeLimit = maxiterationtime
        borderModel.Params.MIPgap = 0.01
        borderModel.setParam( 'OutputFlag', False )

        borderModel.modelSense = GRB.MINIMIZE
        borderModel.optimize(mycallback)
        
        global status
        status= borderModel.status
        
        global MIPgap 
        MIPgap = borderModel.MIPGap
        
        global solcount
        solcount = borderModel.SolCount
        
        global run_time
        run_time = borderModel.runtime
        
        #update the assignvector for each edge
        for e in edgelist:
            for m in range(G.number_of_edges(e[0],e[1])):
                edge = (e[0],e[1],m)
                for k in range(NumberOfContractors):
                    G.edges[edge]['assignvector'][k] = assignVariables[edge][k].x
        
        
    except GurobiError as err:
        print('Error code ' + str(err.errno) + ": " + str(err))
#    
    except AttributeError:
        print('Encountered an attribute error')
    
def mycallback(model,where):
    global objvaluevec
    global boundvec
    global timevec 
    
     
    if where == GRB.Callback.MIP:
            timevec.append(model.cbGet(GRB.Callback.RUNTIME))
            objvaluevec.append(model.cbGet(GRB.Callback.MIP_OBJBST))
            boundvec.append(model.cbGet(GRB.Callback.MIP_OBJBND))

#assign all weights to edges dependent on the weightsetting      
def createweights(weightsetting):
    #Create set attributes
    for e in edgelist:
        for multiplicity in range(G.number_of_edges(e[0],e[1])):
            assignvector = np.zeros(NumberOfContractors)
            k = randint(0,NumberOfContractors-1)
            assignvector[k] = 1         
            nx.set_edge_attributes(G, {(e[0],e[1],multiplicity):{'start':e[0],'end':e[1],'assignvector':assignvector}})
    
    ratio_mu_sigma = sigma/mu
    for e in edgelist:
        for m in range(G.number_of_edges(e[0],e[1])):
            edge = (e[0],e[1],m)
            TPAvector = []
            PPAvector = []
            if weightsetting == 1:
                for k in range(NumberOfContractors):
                    TPAvector.append(np.random.normal(mu,sigma,1))
                    PPAvector.append(np.random.normal(mu,sigma,1))
            if weightsetting == 2:
                for k in range(NumberOfContractors):
                    a = np.random.normal(mu,sigma,1)
                    TPAvector.append(a)
                    PPAvector.append(np.random.normal(a,sigma,1))
            if weightsetting == 3:
                length = G.edges[edge]['length']
                roadtype = G.edges[edge]['highway']
                if type(roadtype) == list:
                    roadtype = 'overig'
                multi_factor = 0.1
                if roadtype in roadtype_to_debris:
                    multi_factor = roadtype_to_debris[roadtype]
                for k in range(NumberOfContractors):
                    a = np.random.normal(length*multi_factor,ratio_mu_sigma*length*multi_factor,1)
                    TPAvector.append(a/averagelength)
                    PPAvector.append(np.random.normal(mu,sigma,1))
            if weightsetting ==4:
                length = G.edges[edge]['length']
                roadtype = G.edges[edge]['highway']
                if type(roadtype) == list:
                    roadtype = 'overig'
                multi_factor = 0.1
                if roadtype in roadtype_to_debris:
                    multi_factor = roadtype_to_debris[roadtype]
                for k in range(NumberOfContractors):
                    a = np.random.normal(length*multi_factor,ratio_mu_sigma*length*multi_factor,1)
                    TPAvector.append(a/averagelength)
                    PPAvector.append(np.random.normal(a,ratio_mu_sigma*length*multi_factor,1)/averagelength)
            if weightsetting ==5:
                for k in range(NumberOfContractors):
                    a = np.random.normal(mu,sigma,1)
                    b = np.random.normal(mu,sigma,1)
                    factor = Contractor[k].k_factor
                    TPAvector.append(a/factor)
                    PPAvector.append(b/factor)
            if weightsetting ==6:
                for k in range(NumberOfContractors):
                    a = np.random.normal(mu,sigma,1)
                    b = np.random.normal(a,sigma,1)
                    factor = Contractor[k].k_factor
                    TPAvector.append(a/factor)
                    PPAvector.append(b/factor)
            if weightsetting ==7:
                length = G.edges[edge]['length']
                roadtype = G.edges[edge]['highway']
                if type(roadtype) == list:
                    roadtype = 'overig'
                multi_factor = 0.1
                if roadtype in roadtype_to_debris:
                    multi_factor = roadtype_to_debris[roadtype]
                for k in range(NumberOfContractors):
                    a = np.random.normal(length*multi_factor,ratio_mu_sigma*length*multi_factor,1)/averagelength
                    b = np.random.normal(mu,sigma,1)
                    factor = Contractor[k].k_factor
                    TPAvector.append(a/factor)
                    PPAvector.append(b/factor)
            if weightsetting ==8:
                length = G.edges[edge]['length']
                roadtype = G.edges[edge]['highway']
                if type(roadtype) == list:
                    roadtype = 'overig'
                multi_factor = 0.1
                if roadtype in roadtype_to_debris:
                    multi_factor = roadtype_to_debris[roadtype]
                for k in range(NumberOfContractors):
                    a = np.random.normal(length*multi_factor,ratio_mu_sigma*length*multi_factor,1)
                    b = np.random.normal(a,ratio_mu_sigma*length*multi_factor,1)
                    factor = Contractor[k].k_factor
                    TPAvector.append(a/(factor*averagelength))
                    PPAvector.append(b/(factor*averagelength))
            nx.set_edge_attributes(G, {(e[0],e[1],m):{'TPAvector':TPAvector,'PPAvector':PPAvector}})
    
    
    
    #create TPAmatrix and PPAmatrix
    global TPAmatrix
    global PPAmatrix
    TPAmatrix = {}
    PPAmatrix = {}
    for e in edgelist:
        for multiplicity in range(G.number_of_edges(e[0],e[1])):
            TPAmatrix[(e[0],e[1],multiplicity)] = []
            PPAmatrix[(e[0],e[1],multiplicity)] = []
            TPAmatrix[(e[0],e[1],multiplicity)] = G.edges[(e[0],e[1],multiplicity)]['TPAvector']
            PPAmatrix[(e[0],e[1],multiplicity)] = G.edges[(e[0],e[1],multiplicity)]['PPAvector']
    
                
#function loading City street map as a graph from Open Street Map           
def createcity(name,Polygon):    
    global NumberOfEdges
    global Contractor
    global G
    global averagelength
       
    #download street map
    G = ox.graph_from_place(name,network_type='drive',which_result=Polygon)
    G = G.to_undirected(reciprocal=False,as_view=False)
    
    #create edgelist
    global edgelist
    edgelist = list(G.edges())
    edgelist = list(dict.fromkeys(edgelist))
    
    #remove selfloops
    for e in edgelist:
        for m in range(G.number_of_edges(e[0],e[1])):
            if e[0] == e[1]:
                G.remove_edge(e[0],e[1])
    
    #update edgelist
    edgelist = list(G.edges())
    edgelist = list(dict.fromkeys(edgelist))
    
    #update number of edges
    NumberOfEdges = G.number_of_edges()
    
    #determine neigbouring edges
    for e in edgelist:
        for m in range(G.number_of_edges(e[0],e[1])):
            edge = (e[0],e[1],m)
            edgereverse = (e[1],e[0],m)
            adjacentedge_vec = []
            for n in [e[0],e[1]]:
                bordernodes = list(G[n])
                for n1 in bordernodes:
                    for m2 in range(G.number_of_edges(n,n1)):
                        adjacentedge = (n,n1,m2)
                        adjacentedgereverse = (n1,n,m2)
                        if adjacentedge != edge and adjacentedge != edgereverse and adjacentedge not in adjacentedge_vec and adjacentedgereverse not in adjacentedge_vec:
                            adjacentedge_vec.append(adjacentedge)
            G.edges[edge]['adjacent_edges'] = adjacentedge_vec
                
    
    #Create node attributes
    for n in G.nodes():
        nx.set_node_attributes(G, {n:{'border':0}})    
    
    #Create contractors
    Contractor = []
    for k in range(NumberOfContractors):
        color = colorvector[k]
        TPA = 0
        PPA = 0
        k_factor = k_factor_vec[k]
        Contractor.append(Contr(color,TPA,PPA,k_factor))
    
    #Calculate average edge length
    total = 0
    for e in edgelist:
        for m in range(G.number_of_edges(e[0],e[1])):
            edge = (e[0],e[1],m)
            total += G.edges[edge]['length']
    averagelength = total/NumberOfEdges
        
```
        
            
```python        
################################################################
##############################set parameters####################
################################################################

#weight parameters
mu = 5
sigma = 1
roadtype_to_debris = {'motorway':0.1,'primary':0.2,'secondary':0.3,'tertiary':0.4,'residential':1,'other':0.1}

#plot parameter
colorvector = ['rgb(255,0,0)','rgb(0,255,0)','rgb(0,0,255)','rgb(255,255,0)','rgb(255,0,255)','rgb(0,255,255)','rgb(0,0,0)','rgb(190,190,190)','rgb(100,100,100)','rgb(40,40,100)']

#Gurobi parameters
maxiterationtime = 3600*5 #5 hours running time for the brute force algorithm
gurobi_opt_code ={2:'optimal',3:'infeasible',9:'timelimit reached'}

#GAPalgorithm parameters
infty = 100000000
````

```python       
#########################################################################
#############################run program#################################
#########################################################################


#read csv file
Seed_read = []
Cities_read =[]
Polygon_read = []
Contractors_read = []
Setting_read = []

with open('Third_run.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=';')
    for row in readCSV:
        Seed_read.append(row[0])
        Cities_read.append(row[1])
        Polygon_read.append(row[2])
        Contractors_read.append(row[3])
        Setting_read.append(row[4])
        
Seed = int(sys.argv[1])
City = Cities_read[Seed]
Polygon = int(Polygon_read[Seed])
NumberOfContractors = int(Contractors_read[Seed])
weightsetting = int(Setting_read[Seed])
seedvalue = Seed 

#debris setting parameter
k_factor_vec = np.linspace(1,NumberOfContractors,NumberOfContractors)

#create city
createcity(City,Polygon)
seed(seedvalue)
createweights(weightsetting)
updatevalues()
 
#optimize time
TPAspreadfunc()
updatevalues()  
TPA_opt = TPA
    
#optimize profit
PPAspreadfunc()
updatevalues()  
PPA_opt = PPA
    
#determine range of time values serving as bounds in the following Pareto Front Optimization
boundPPA_TPA_opt(PPA_opt*0.99)
updatevalues()  
TPA_max = TPA

#Pareto Front Optimization
TPAvec = []
PPAvec = []
TPAratiovec = []
PPAratiovec = []
for TPAvalue in np.linspace(TPA_opt*1.01,TPA_max,10):
    boundTPA_PPA_opt(TPAvalue)
    updatevalues()
    TPAvec.append(TPA)
    PPAvec.append(PPA)
    TPAratiovec.append(2-TPA/TPA_opt)
    PPAratiovec.append(PPA/PPA_opt)

#Find the solution with the highest min(TPA,PPA)
maximum = -1
for j in range(len(TPAratiovec)):
    if min(TPAratiovec[j],PPAratiovec[j]) > maximum:
        maximum = min(TPAratiovec[j],PPAratiovec[j])
        optimalratio = [PPAratiovec[j],TPAratiovec[j]]
        optimal = [PPAvec[j],TPAvec[j]]

#for three alpha's solve the instance in two ways: with the brute force algorithm and with the GAP algorithm
for alpha in [0.4,0.2,0.1]:
    TPAbound = optimal[1]*(1+alpha)
    PPAbound = optimal[0]*(1-alpha)

    #we run the brute force algorithm
    seed(seedvalue)
    createweights(weightsetting)
    updatevalues()
    objvaluevec = [] #vector to be filled  with niceness during optimization
    boundvec= [] #vector to be filled with lower bound during optimization
    timevec = [] #vector to be filled with time values where the above are achieved during optimization
    optimize_border(TPAbound,PPAbound)
    updatevalues()
    update_components()

    #we paste the results of the brute force algorithm
    with open('brute_force_outputv3.csv', 'a',newline='') as output:
        create_output = csv.writer(output, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        create_output.writerow([seedvalue, City, NumberOfContractors,weightsetting,alpha,G.number_of_edges(),niceness,len(Component_vec),MIPgap,gurobi_opt_code[status],objvaluevec,boundvec,solcount,run_time,timevec])

    #run the GAP algorithm, starting with the "start" algorithm
    seed(seedvalue)
    createweights(weightsetting)
    updatevalues()
    randomrestrict = 1
    update_swapvalues()
    update_components()
    
    nicenessvec = [niceness] #vector storing the niceness
    componentvec = [len(Component_vec)] #vector storing the number of components
    nicenesstimevec = [0] #vector storing the time the niceness is achieved
    componenttimevec = [0] #vector storing the time the component number is achieved
    start = timer()
    
    for j in range(5):
        for swapmethod in ['regular','restrictedswapvalue']:
            GAPalgorithm(TPAbound,PPAbound,swapmethod)
            updatevalues()
            update_swapvalues()
            nicenessvec.append(niceness)
            nicenesstimevec.append(timer()-start)
    
    #we now run the "ending" GAP algorithm
    update_components()
    componentvec.append(len(Component_vec))
    componenttimevec.append(timer()-start)
    
    for j in range(10):
        oldniceness = niceness
        for swapmethod in ['regular','restricted']:
                GAPalgorithm_comp(TPAbound,PPAbound,swapmethod)
                updatevalues()
                update_components()
                componentvec.append(len(Component_vec))
                componenttimevec.append(timer()-start)
                nicenessvec.append(niceness)
                nicenesstimevec.append(timer()-start)            
        if niceness >= oldniceness:
            break
    
    #we paste the results of the GAP algorithm
    with open('GAPoutputv3.csv', 'a',newline='') as output:
        create_output = csv.writer(output, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        create_output.writerow([seedvalue, City, NumberOfContractors,weightsetting,alpha,G.number_of_edges(),niceness,len(Component_vec),nicenessvec,componentvec,nicenesstimevec,componenttimevec])
   
```
