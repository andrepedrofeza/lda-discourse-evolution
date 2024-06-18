import math
import pandas as pd
import numpy as np
from typing import List


class MetricsCalculator:
    def __calculateCentralDiscourseFormula(
        self,
        values
    ) -> List[float]:
        return np.mean(np.array(values.tolist()), axis=0).tolist()
    
    def calculateCentralDiscourse(
        self,
        df: pd.DataFrame,
        group_cols: str|List[str]|None
    ) -> pd.Series:
        if group_cols:
            return df.groupby(group_cols)['td'].agg(self.__calculateCentralDiscourseFormula)
        
        return self.__calculateCentralDiscourseFormula(df['td'])

    def __calculateDiversityFormula(
        self,
        values,
    ) -> float:
        diversity = 0.0

        for i in range(len(values)):
            diversity += values[i] * math.log(values[i])

        return -1 * diversity
        
    def calculateDiversity(
        self,
        df: pd.DataFrame,
        group_cols: str|List[str],
    ) -> pd.Series:
        cd = self.calculateCentralDiscourse(df, group_cols)

        return cd.apply(self.__calculateDiversityFormula)

    def __calculateCohesionFormula(
        self,
        values,
        cd
    ) -> float:
        l1_norm = np.linalg.norm(
            (np.array(values.tolist()) - np.array(cd)),
            ord=1,
            axis=1
        )

        cohesion = (-1 / len(values)) * np.sum(l1_norm)

        return cohesion

    def calculateCohesion(
        self,
        df: pd.DataFrame,
        group_cols: str|List[str]
    ) -> pd.Series:
        cd = self.__calculateCentralDiscourseFormula(df['td'])

        return df.groupby(group_cols)['td'].agg(self.__calculateCohesionFormula, cd=cd)
    
    def __calculateDissonanceFormula(
        self,
        values,
        cd,
        weight_vector
    ) -> float:
        return np.sqrt(np.sum(np.multiply(np.square(np.subtract(values, cd)), weight_vector)))

    def calculateDissonance(
        self,
        df: pd.DataFrame,
        entity_col: str,
        entity_group_col: str
    ) -> pd.Series:
        entity_discourse = self.calculateCentralDiscourse(df, [entity_col, entity_group_col]).to_frame()
        group_discouse = self.calculateCentralDiscourse(df, [entity_group_col]).to_frame()

        def calculateWeightVector(td):
            indexed_values = [(value, index) for index, value in enumerate(td)]
            sorted_values = sorted(indexed_values, key=lambda x: x[0])
            sorted_indexes = [0] * len(sorted_values)
            
            for index, value in enumerate(sorted_values):
                sorted_indexes[value[1]] = index + 1
            
            weight_vector = np.log(np.array(sorted_indexes)) + 1

            return weight_vector
        
        group_discouse['weight_vector'] = group_discouse['td'].apply(calculateWeightVector)

        discourses = entity_discourse.join(group_discouse, rsuffix='_group')
        
        return discourses.apply(lambda row: self.__calculateDissonanceFormula(row['td'], row['td_group'], row['weight_vector']), axis=1)

    def __calculateCoherenceFormula(
        self,
        values
    ) -> float:
        return -1 * np.mean(np.var(np.stack(values), axis=0))

    def calculateCoherence(
        self,
        df: pd.DataFrame,
        group_cols: str|List[str]
    ) -> pd.Series:
        return df.groupby(group_cols)['td'].agg(self.__calculateCoherenceFormula)
