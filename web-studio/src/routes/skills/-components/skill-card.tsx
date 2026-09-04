import {
  ChevronRightIcon,
  LoaderCircleIcon,
  Share2Icon,
  SparklesIcon,
} from 'lucide-react'
import { useTranslation } from 'react-i18next'

import { Badge } from '#/components/ui/badge'
import { Button } from '#/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '#/components/ui/card'

import { SKILL_SCOPE_ICONS } from './skill-scope-tabs'
import type { SkillScope } from './skill-scope-tabs'

export type SkillCardItem = {
  description: string
  name: string
  scope: SkillScope
  uri: string
}

export function SkillCard({
  isSharing,
  onOpen,
  onShare,
  skill,
}: {
  isSharing: boolean
  onOpen: () => void
  onShare: () => void
  skill: SkillCardItem
}) {
  const { t } = useTranslation('skillsPage')
  const ScopeIcon = SKILL_SCOPE_ICONS[skill.scope]

  return (
    <Card size="sm" className="h-full transition-colors hover:bg-muted/35">
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <SparklesIcon className="size-4" />
            </div>
            <CardTitle className="truncate">{skill.name}</CardTitle>
          </div>
          <Badge variant="outline" className="gap-1 font-normal">
            <ScopeIcon />
            {t(`scopes.${skill.scope}`)}
          </Badge>
        </div>
        {skill.description ? (
          <CardDescription className="line-clamp-2 pt-1 leading-5">
            {skill.description}
          </CardDescription>
        ) : null}
      </CardHeader>
      <CardContent className="mt-auto">
        <div className="flex items-center justify-between gap-3">
          <code className="min-w-0 flex-1 truncate text-xs text-muted-foreground">
            {skill.uri}
          </code>
          <div className="flex shrink-0 items-center gap-1">
            {skill.scope === 'user' ? (
              <Button
                type="button"
                variant="outline"
                size="xs"
                disabled={isSharing}
                aria-label={t('shareSkill', { name: skill.name })}
                onClick={onShare}
              >
                {isSharing ? (
                  <LoaderCircleIcon className="animate-spin" />
                ) : (
                  <Share2Icon />
                )}
                {isSharing ? t('sharing') : t('share')}
              </Button>
            ) : null}
            <Button
              type="button"
              variant="ghost"
              size="xs"
              aria-label={t('viewDetail', { name: skill.name })}
              onClick={onOpen}
            >
              {t('detail')}
              <ChevronRightIcon />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
