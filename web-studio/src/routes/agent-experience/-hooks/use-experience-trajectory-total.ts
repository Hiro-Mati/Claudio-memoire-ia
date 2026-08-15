import { useQuery } from '@tanstack/react-query'

import { fetchTrajectories } from '../-lib/api'

/**
 * Lazy per-experience applied-trajectory count for list rows.
 *
 * Uses `limit=1` so the response only carries the total; results are cached
 * per identity scope to avoid re-counting on every render.
 */
export function useExperienceTrajectoryTotal(
  experienceUri: string,
  identityScopeKey: string,
) {
  const query = useQuery({
    enabled: Boolean(experienceUri),
    queryFn: ({ signal }) =>
      fetchTrajectories({
        experienceUri,
        limit: 1,
        offset: 0,
        signal,
      }),
    queryKey: [
      'agent-experience-trajectory-total',
      identityScopeKey,
      experienceUri,
    ],
    staleTime: 60_000,
  })

  return {
    isPending: query.isPending && Boolean(experienceUri),
    total: query.data?.total,
  }
}
