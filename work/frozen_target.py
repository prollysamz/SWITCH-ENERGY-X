"""Freeze both legacy normalizations and all scored target coefficients."""
import numpy as np
from build_hypotheses import components, expected_drawdown

WEIGHTS=np.array([.25,.20,.20,.15,.10,.10])


def freeze(train_blocks,test_blocks,hypothesis,plane):
    ctr,cte=components(train_blocks),components(test_blocks)
    allc=np.concatenate([ctr,cte],axis=1)
    out={'component_weights':WEIGHTS.tolist(),'target_mean':3.9,'original_sd':4.4,
         'generation_sd':hypothesis['assumptions']['generation_sd'],'new_effect_sd':2.,
         'heat_beta':.004213,'heat_threshold':40.,'temperature_reference':25.,
         'normalization':hypothesis['normalization'],'plane_weights':plane['weights']}
    for name,c in [('original',allc),('hypothesis',ctr)]:
        mu,sd=c.mean(axis=1),c.std(axis=1)
        blend=WEIGHTS@((c-mu[:,None])/sd[:,None])
        out[name]={'component_mean':mu.tolist(),'component_sd':sd.tolist(),
                   'blend_mean':float(blend.mean()),'blend_sd':float(blend.std())}
    return out


def predict(blocks,frozen):
    c=components(blocks)
    def blend(name):
        norm=frozen[name]
        raw=np.asarray(frozen['component_weights'])@(
            (c-np.asarray(norm['component_mean'])[:,None])/np.asarray(norm['component_sd'])[:,None])
        return (raw-norm['blend_mean'])/norm['blend_sd']
    g=blend('hypothesis')
    original=frozen['target_mean']+frozen['original_sd']*blend('original')
    def standardize(value,key):
        norm=frozen['normalization'][key]
        return (value-norm['mean']-norm['slope']*g)/norm['sd']
    battery=standardize(-expected_drawdown(blocks),'battery')
    panel=(1-blocks['E_tau'])/frozen['heat_beta']+frozen['temperature_reference']
    heat=standardize(-np.maximum(panel-frozen['heat_threshold'],0),'heat')
    combined=standardize(.8*battery+.6*heat,'combined')
    base=frozen['target_mean']+frozen['generation_sd']*g
    b=base+frozen['new_effect_sd']*battery
    c=base+frozen['new_effect_sd']*combined
    w1,w2=frozen['plane_weights']
    return b+w1*(c-b)+w2*(original-b)
