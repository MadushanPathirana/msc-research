class User:    
    def user_dataset(df,user):  
        df[df.user_id==user].correct.value_counts()
    
        df['correctBinary']=df.correct*1
    
        df.atom_id=df.atom_id.astype(str)
    
        userDf=df[df.user_id==user]
    
        userDf=userDf.sort_values(by='interaction_end_time')
    
        atomIndex={atom:i for i,atom in enumerate( userDf.atom_id)}
    
        #loIndex={lo:i for i,lo in enumerate( userDf.learning_objective_name.unique())}
    
        #userDf['loIndex']=userDf['learning_objective_name'].apply(lambda x: loIndex[x])
        userDf['atomIndex']=np.arange(userDf.shape[0])
    
        atomsPerLO=userDf.groupby('loIndex')['atom_id'].count().to_dict()
        userDf['atomsPerLO']=userDf['loIndex'].apply(lambda x: atomsPerLO[x])
    
        edgeMap=userDf[['loIndex','atomIndex']]
    
        y=userDf.correctBinary.astype(int).to_numpy()
    
        #loMapping=df2[(df2.source_lo_title.isin(loIndex.keys())) | (df2.dest_lo_title.isin(loIndex.keys()))]
    
        #loMapping['source_index']=loMapping.source_lo_title.map(loIndex)
        #loMapping['dest_index']=loMapping.dest_lo_title.map(loIndex)
    
        #loEdgeMapping=loMapping[['source_index','dest_index']].reset_index(drop=True)
    
        #loEdgeIndex=loEdgeMapping.values.transpose()
    
        loAtomEdgeIndex=edgeMap.values.transpose()
    
        atomAtom=edgeMap.reset_index(drop=True)[['atomIndex']]
    
        atomAtom['nextAtom']=atomAtom.atomIndex.shift(-1)
    
        atomAtom=atomAtom.dropna()
    
        atomEdgeIndex=atomAtom.values.transpose()
    
        atomFeatures=userDf.difficulty.to_numpy().reshape(-1,1)
    
        loFeatures=np.array(list(atomsPerLO.values())).reshape(-1,1)
        return  atomFeatures,loFeatures,loAtomEdgeIndex,atomEdgeIndex,y