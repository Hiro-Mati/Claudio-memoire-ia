import * as React from 'react'
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { Link, createFileRoute } from '@tanstack/react-router'
import {
  ArrowLeftIcon,
  BarChart3Icon,
  BrainCircuitIcon,
  ClipboardIcon,
  FileTextIcon,
  LoaderCircleIcon,
  RefreshCwIcon,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { toast } from 'sonner'

import { Button } from '#/components/ui/button'
import { ButtonGroup } from '#/components/ui/button-group'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '#/components/ui/card'
import { Separator } from '#/components/ui/separator'
import { useAppConnection } from '#/hooks/use-app-connection'
import { copyTextToClipboard } from '#/lib/clipboard'
import { isOvClientError } from '#/lib/ov-client'

import { OutcomeDistribution } from './-components/outcome-distribution'
import { TrajectoryList } from './-components/trajectory-list'
import { TrajectoryPreviewSheet } from './-components/trajectory-preview-sheet'
import {
  fetchContent,
  fetchOutcomeDistribution,
  fetchTrajectories,
} from './-lib/api'
import { getExperienceDisplayName, resolveTimeRange } from './-lib/experience'
import type {
  TimeRange,
  TimeRangePreset,
  TrajectoryItem,
  TrajectoryPage,
} from './-lib/types'

export const Route = createFileRoute('/agent-experience/$experienceUri')({
  component: ExperienceDetailRoute,
})

const TIME_RANGE_PRESETS: readonly TimeRangePreset[] = ['all', '7d', '30d']

function getErrorMessage(error: unknown): string {
  if (isOvClientError(error) || error instanceof Error) {
    return error.message
  }
  return String(error)
}

function ExperienceDetailRoute() {
  const { t, i18n } = useTranslation('agentExperiencePage')
  const { identityScopeKey } = useAppConnection()
  const params = Route.useParams()
  // The router percent-encodes path params when building hrefs and decodes
  // them again on match, so `params.experienceUri` is the raw `viking://` URI.
  const experienceUri = params.experienceUri

  const [timeRangePreset, setTimeRangePreset] =
    React.useState<TimeRangePreset>('all')
  const timeRange = React.useMemo<TimeRange>(
    () => resolveTimeRange(timeRangePreset),
    [timeRangePreset],
  )
  const [selectedTrajectory, setSelectedTrajectory] =
    React.useState<TrajectoryItem | null>(null)

  const contentQuery = useQuery({
    queryFn: ({ signal }) => fetchContent(experienceUri, signal),
    queryKey: ['agent-experience-content', identityScopeKey, experienceUri],
    staleTime: 60_000,
  })

  const outcomeQuery = useQuery({
    queryFn: ({ signal }) =>
      fetchOutcomeDistribution({ experienceUri, signal, timeRange }),
    queryKey: [
      'agent-experience-outcomes',
      identityScopeKey,
      experienceUri,
      timeRange.preset,
      timeRange.startDate,
      timeRange.endDate,
    ],
  })

  const trajectoriesQuery = useInfiniteQuery<
    TrajectoryPage,
    Error,
    { pages: TrajectoryPage[]; pageParams: number[] },
    readonly unknown[],
    number
  >({
    initialPageParam: 0,
    getNextPageParam: (lastPage) =>
      lastPage.hasMore ? lastPage.offset + lastPage.items.length : undefined,
    queryFn: ({ pageParam, signal }) =>
      fetchTrajectories({
        experienceUri,
        offset: pageParam,
        signal,
        timeRange,
      }),
    queryKey: [
      'agent-experience-trajectories',
      identityScopeKey,
      experienceUri,
      timeRange.preset,
      timeRange.startDate,
      timeRange.endDate,
    ],
  })

  const trajectories =
    trajectoriesQuery.data?.pages.flatMap((page) => page.items) ?? []
  const trajectoryTotal = trajectoriesQuery.data?.pages[0]?.total ?? 0
  const hasMoreTrajectories = Boolean(
    trajectoriesQuery.hasNextPage && !trajectoriesQuery.isFetchingNextPage,
  )
  const experienceName = getExperienceDisplayName(experienceUri)

  const handleCopyUri = () => {
    void copyTextToClipboard(experienceUri)
      .then(() => {
        toast.success(t('detail.copied'))
      })
      .catch(() => {
        toast.error(t('detail.copyFailed'))
      })
  }

  return (
    <div className="flex w-full min-w-0 flex-col gap-5">
      <header className="grid gap-3">
        <div>
          <Button
            render={<Link to="/agent-experience" />}
            nativeButton={false}
            size="xs"
            variant="ghost"
            className="-ml-2 text-muted-foreground"
          >
            <ArrowLeftIcon className="size-3.5" />
            {t('detail.back')}
          </Button>
        </div>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <BrainCircuitIcon className="size-4.5" />
            </div>
            <div className="min-w-0">
              <h1
                className="truncate text-xl font-semibold tracking-tight"
                title={experienceName}
              >
                {experienceName}
              </h1>
              <div className="flex items-center gap-1.5">
                <code
                  className="min-w-0 truncate text-xs text-muted-foreground"
                  title={experienceUri}
                >
                  {experienceUri}
                </code>
                <Button
                  type="button"
                  aria-label={t('detail.copyUri')}
                  size="icon-xs"
                  variant="ghost"
                  onClick={handleCopyUri}
                >
                  <ClipboardIcon className="size-3.5" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="grid min-w-0 items-start gap-4 xl:grid-cols-[minmax(0,5fr)_minmax(0,4fr)]">
        <Card size="sm">
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <div className="flex min-w-0 items-center gap-2">
                <FileTextIcon className="size-4 shrink-0 text-muted-foreground" />
                <CardTitle className="truncate text-base">
                  {t('detail.contentTitle')}
                </CardTitle>
              </div>
              <Button
                type="button"
                aria-label={t('refresh')}
                size="icon-xs"
                variant="ghost"
                disabled={contentQuery.isFetching}
                onClick={() => void contentQuery.refetch()}
              >
                <RefreshCwIcon
                  className={
                    contentQuery.isFetching ? 'animate-spin' : undefined
                  }
                />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="px-0">
            {contentQuery.isLoading ? (
              <div className="flex min-h-64 items-center justify-center gap-2 px-6 text-sm text-muted-foreground">
                <LoaderCircleIcon className="size-4 animate-spin" />
                {t('detail.contentLoading')}
              </div>
            ) : contentQuery.isError ? (
              <div className="grid min-h-64 place-items-center gap-2 px-6 text-center">
                <p className="font-medium">{t('detail.contentLoadFailed')}</p>
                <p className="max-w-md text-sm text-muted-foreground">
                  {getErrorMessage(contentQuery.error)}
                </p>
                <Button
                  type="button"
                  size="xs"
                  variant="outline"
                  onClick={() => void contentQuery.refetch()}
                >
                  {t('refresh')}
                </Button>
              </div>
            ) : (
              <div className="max-h-[70vh] overflow-y-auto px-5 pb-5">
                <div className="prose prose-sm max-w-none break-words dark:prose-invert dark:prose-pre:bg-muted-foreground/20">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {contentQuery.data || ''}
                  </ReactMarkdown>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card size="sm">
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="grid min-w-0 gap-1">
                <div className="flex min-w-0 items-center gap-2">
                  <BarChart3Icon className="size-4 shrink-0 text-muted-foreground" />
                  <CardTitle className="truncate text-base">
                    {t('detail.analysisTitle')}
                  </CardTitle>
                </div>
                <CardDescription>
                  {t('detail.analysisDescription')}
                </CardDescription>
              </div>
              <ButtonGroup>
                {TIME_RANGE_PRESETS.map((preset) => (
                  <Button
                    key={preset}
                    type="button"
                    aria-pressed={timeRangePreset === preset}
                    variant={timeRangePreset === preset ? 'secondary' : 'ghost'}
                    size="xs"
                    onClick={() => setTimeRangePreset(preset)}
                  >
                    {t(
                      preset === 'all'
                        ? 'detail.rangeAll'
                        : preset === '7d'
                          ? 'detail.range7d'
                          : 'detail.range30d',
                    )}
                  </Button>
                ))}
              </ButtonGroup>
            </div>
          </CardHeader>
          <CardContent className="grid gap-5 px-5 pb-5">
            <section className="grid gap-3">
              <h3 className="text-sm font-medium">
                {t('detail.outcomeTitle')}
              </h3>
              {outcomeQuery.isLoading ? (
                <div className="flex min-h-16 items-center gap-2 text-sm text-muted-foreground">
                  <LoaderCircleIcon className="size-4 animate-spin" />
                  {t('detail.loadingMore')}
                </div>
              ) : outcomeQuery.isError ? (
                <div className="grid min-h-16 place-items-center gap-2 text-center">
                  <p className="text-sm text-muted-foreground">
                    {getErrorMessage(outcomeQuery.error)}
                  </p>
                  <Button
                    type="button"
                    size="xs"
                    variant="outline"
                    onClick={() => void outcomeQuery.refetch()}
                  >
                    {t('refresh')}
                  </Button>
                </div>
              ) : outcomeQuery.data ? (
                <OutcomeDistribution
                  distribution={outcomeQuery.data.distribution}
                />
              ) : null}
            </section>

            <Separator />

            <section className="grid gap-3">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-medium">
                  {t('detail.trajectoriesTitle')}
                </h3>
                <span className="text-xs text-muted-foreground">
                  {t('detail.rangeUtcHint')}
                </span>
              </div>
              <TrajectoryList
                error={trajectoriesQuery.error}
                hasMore={hasMoreTrajectories}
                isLoading={trajectoriesQuery.isLoading}
                isLoadingMore={trajectoriesQuery.isFetchingNextPage}
                items={trajectories}
                language={i18n.language}
                onLoadMore={() => void trajectoriesQuery.fetchNextPage()}
                onRetry={() => void trajectoriesQuery.refetch()}
                onSelect={setSelectedTrajectory}
                total={trajectoryTotal}
              />
            </section>
          </CardContent>
        </Card>
      </div>

      <TrajectoryPreviewSheet
        language={i18n.language}
        onClose={() => setSelectedTrajectory(null)}
        trajectory={selectedTrajectory}
      />
    </div>
  )
}
